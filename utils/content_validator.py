#utils/content_validator.py
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv
from utils.session_logger import get_session_logger
from config import LOG_DIR, OUTPUT_DIR
from functools import lru_cache
from PIL import Image
from io import BytesIO
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import spacy

load_dotenv()

# Load spaCy model for better role parsing
nlp = spacy.load("en_core_web_sm", disable=["parser"])

class ValidationResponse(BaseModel):
    decision: str = Field(description="Either 'Yes' or 'No'")
    explanation: str = Field(description="Brief explanation of the decision (1-2 sentences)")
    confidence: int = Field(description="Confidence score from 0 to 100")

@lru_cache(maxsize=1000)
def cached_extract_text(file_path, file_type):
    """Cache extracted text from a file based on its type."""
    session_logger = get_session_logger(LOG_DIR, session_id=os.path.basename(file_path))
    try:
        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            text = " ".join(doc.page_content for doc in documents)
        elif file_type == "docx":
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            text = " ".join(doc.page_content for doc in documents)
        elif file_type == "xlsx":
            loader = UnstructuredExcelLoader(file_path, mode="elements")
            documents = loader.load()
            text = " ".join(doc.page_content for doc in documents)
        elif file_type in ["jpg", "jpeg", "png"]:
            ocr = ocr_predictor(pretrained=True)
            doc = DocumentFile.from_images(file_path)
            result = ocr(doc)
            text = ""
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            text += word.value + " "
            text = text.strip()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        if not text.strip():
            session_logger.log(
                component="Document Validation",
                message=f"Extract text from {os.path.basename(file_path)}",
                decision="Rejected",
                reason="Extracted text is empty",
                level="ERROR",
                context={"file": os.path.basename(file_path), "file_type": file_type}
            )
            return ""
        return text[:10000]
    except Exception as e:
        session_logger.log(
            component="Document Validation",
            message=f"Extract text from {os.path.basename(file_path)}",
            decision="Rejected",
            reason=f"Extraction error: {str(e)}",
            level="ERROR",
            context={"file": os.path.basename(file_path), "file_type": file_type}
        )
        return ""

def parse_roles_with_nlp(text):
    """Extract roles using spaCy to distinguish names from roles."""
    doc = nlp(text)
    # Identify roles by looking for non-person entities or specific keywords
    role_keywords = {"director", "manager", "vp", "ceo", "cfo", "cto", "coo", "chair", "analyst", "engineer", "recruiter", "strategist"}
    role = []
    for token in doc:
        if token.lower_ in role_keywords or (token.pos_ in ["NOUN", "ADJ"] and token.ent_type_ != "PERSON"):
            role.append(token.text)
    return " ".join(role).strip() or "Unknown"

def detect_lines(image_path):
    """Detect lines in the image using HoughLinesP, treating them as connectors."""
    image = cv2.imread(image_path)
    if image is None:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Enhance line detection with adaptive thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=10)
    connectors = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            connectors.append({"start": [x1, y1], "end": [x2, y2], "length": np.sqrt((x2 - x1)**2 + (y2 - y1)**2)})
    return connectors

def cluster_boxes(positions):
    """Group nearby text boxes using DBSCAN with adjusted eps."""
    if len(positions) == 0:
        return np.array([])
    db = DBSCAN(eps=100, min_samples=1).fit(positions)  # Increased eps for better separation
    return db.labels_

def process_org_chart_image(file_path, session_id, session_logger):
    """Process an image-based organizational chart using OCR and computer vision."""
    start_time = time.time()
    try:
        # Initialize OCR predictor
        ocr = ocr_predictor(pretrained=True)
        doc = DocumentFile.from_images(file_path)
        result = ocr(doc)

        # Extract text and bounding boxes
        blocks = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    text = ' '.join([w.value for w in line.words]).strip()
                    if not text:
                        continue
                    role = parse_roles_with_nlp(text)
                    x_min, y_min = line.geometry[0]
                    x_max, y_max = line.geometry[1]
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2
                    blocks.append({
                        "text": text,
                        "role": role,
                        "center": [center_x, center_y],
                        "bbox": [x_min, y_min, x_max, y_max],
                        "y_center": center_y  # Store for hierarchy sorting
                    })

        # Cluster text boxes
        positions = np.array([b["center"] for b in blocks])
        labels = cluster_boxes(positions)
        for idx, block in enumerate(blocks):
            block["group"] = int(labels[idx]) if len(labels) > 0 else 0

        # Detect lines
        lines = detect_lines(file_path)

        # Construct hierarchy using vertical positioning and line proximity
        hierarchy = []
        for idx, block in enumerate(blocks):
            related_lines = [
                line for line in lines
                if min(np.linalg.norm(np.array(line["start"]) - np.array(block["center"])),
                       np.linalg.norm(np.array(line["end"]) - np.array(block["center"]))) < 75
            ]
            relationships = []
            for line in related_lines:
                for other_block in blocks:
                    if other_block == block:
                        continue
                    # Check if the other block is connected by the line
                    dist_to_other = min(
                        np.linalg.norm(np.array(line["start"]) - np.array(other_block["center"])),
                        np.linalg.norm(np.array(line["end"]) - np.array(other_block["center"]))
                    )
                    if dist_to_other < 75:
                        # Determine direction based on y-coordinate (higher y = lower in hierarchy)
                        direction = "reports_to" if block["y_center"] > other_block["y_center"] else "parent_of"
                        relationships.append({
                            "to": other_block["text"],
                            "to_role": other_block["role"],
                            "direction": direction,
                            "confidence": 1.0 - (dist_to_other / 75)  # Confidence based on proximity
                        })
            # Sort relationships by confidence
            relationships = sorted(relationships, key=lambda x: x["confidence"], reverse=True)[:3]  # Limit to top 3
            hierarchy.append({
                "name": block["text"],
                "role": block["role"],
                "group": block["group"],
                "bbox": block["bbox"],
                "y_center": block["y_center"],
                "relationships": relationships
            })

        # Combine all text for validation
        text = " ".join([block["text"] for block in blocks]).strip()

        # Save results to outputs folder
        output_filename = f"org_chart_{session_id}_{os.path.basename(file_path)}.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        output_data = {
            "file": os.path.basename(file_path),
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "text": text,
            "hierarchy": hierarchy,
            "lines": lines
        }
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)
            session_logger.log(
                component="Org Chart Processing",
                message=f"Saved org chart interpretation to {output_filename}",
                decision="Accepted",
                reason="JSON file saved successfully",
                level="INFO",
                context={"file": os.path.basename(file_path), "output_path": output_path}
            )
        except Exception as e:
            session_logger.log(
                component="Org Chart Processing",
                message=f"Saving org chart interpretation to {output_filename}",
                decision="Rejected",
                reason=f"Failed to save JSON: {str(e)}",
                level="ERROR",
                context={"file": os.path.basename(file_path), "output_path": output_path}
            )

        return text, hierarchy
    except Exception as e:
        session_logger.log(
            component="Org Chart Processing",
            message=f"Processing {os.path.basename(file_path)}",
            decision="Rejected",
            reason=f"Processing error: {str(e)}",
            level="ERROR",
            context={"file": os.path.basename(file_path)}
        )
        return "", []

def extract_text_from_file(file_path, session_logger=None):
    """Extract text from a file using appropriate LangChain loader or DocTR for images based on extension."""
    start_time = time.time()
    file_extension = os.path.splitext(file_path)[1].lower().lstrip(".")
    file_type = file_extension if file_extension in ["pdf", "docx", "xlsx", "jpg", "jpeg", "png"] else None
    
    if not file_type:
        if session_logger:
            session_logger.log(
                component="Document Validation",
                message=f"Extract text from {os.path.basename(file_path)}",
                decision="Rejected",
                reason=f"Unsupported file extension: {file_extension}",
                level="ERROR",
                context={"file": os.path.basename(file_path)}
            )
        return ""

    text = cached_extract_text(file_path, file_type)
    response_time = (time.time() - start_time) * 1000
    
    if session_logger:
        session_logger.log(
            component="Document Validation",
            message=f"Extract text from {os.path.basename(file_path)}",
            decision="Accepted" if text else "Rejected",
            reason="Text extracted successfully" if text else "File Extraction Error",
            source=f"Text Preview={text[:100]}..." if text else None,
            level="DEBUG",
            context={"file": os.path.basename(file_path), "file_type": file_type, "text_length": len(text), "response_time_ms": int(response_time)}
        )
    
    return text

def validate_document(text, category, session_logger=None):
    """Validate document using LangChain and OpenAI GPT-4o-mini with structured output."""
    if not text:
        if session_logger:
            session_logger.log(
                component="Document Validation",
                message=f"Validate {category}",
                decision="Rejected",
                reason="Empty text",
                level="ERROR",
                context={"category": category}
            )
        return False

    start_time = time.time()
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0
        ).with_structured_output(ValidationResponse)

        if category == "Organizational Chart":
            prompt_template = """
            You are tasked with determining if the provided text, extracted from a document, represents a valid Organizational Chart describing a company’s structure, roles, or hierarchy. A valid Organizational Chart must include at least one name and job title, or a clear hierarchy (e.g., "CEO reports to Board"). Ignore headers, footers, page numbers, or irrelevant text like disclaimers.

            **Instructions:**
            - Return a JSON object with:
              - "decision": "Yes" or "No"
              - "explanation": Brief explanation (1-2 sentences)
              - "confidence": Integer from 0 to 100 indicating confidence
            - If the text is empty, too short (<50 characters), or clearly unrelated (e.g., a contract, meeting notes, or image-based document with no meaningful text), set decision to "No" and confidence to 0.
            - Examples of valid content:
              - "John Doe, CEO; Jane Smith, CFO, reports to CEO"
              - "Marketing Department: Alice Brown, Manager"
            - Examples of invalid content:
              - "Meeting agenda for 2025"
              - "Company logo and address"
              - "Blank page"

            Text: {document_text}
            """
        elif category == "Governance Document":
            prompt_template = """
            You are tasked with determining if the provided text, extracted from a document, represents a valid Governance Document detailing policies, procedures, compliance, or board regulations. A valid Governance Document must include terms like "policy," "procedure," "compliance," "board," or "regulation," and outline rules or frameworks. Ignore headers, footers, page numbers, or unrelated text like advertisements.

            **Instructions:**
            - Return a JSON object with:
              - "decision": "Yes" or "No"
              - "explanation": Brief explanation (1-2 sentences)
              - "confidence": Integer from 0 to 100 indicating confidence
            - If the text is empty, too short (<50 characters), or clearly unrelated (e.g., an org chart, invoice, or employee handbook), set decision to "No" and confidence to 0.
            - Examples of valid content:
              - "Board Policy: All decisions must be approved by a majority"
              - "Compliance Procedure: Annual audits required"
            - Examples of invalid content:
              - "Financial report 2025"
              - "Marketing plan"
              - "Blank page"

            Text: {document_text}
            """
        else:
            if session_logger:
                session_logger.log(
                    component="Document Validation",
                    message=f"Validate {category}",
                    decision="Rejected",
                    reason=f"Invalid category {category}",
                    level="ERROR",
                    context={"category": category}
                )
            return False

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        prompt_text = prompt_template.format(document_text=text[:100] + "...")
        response = chain.invoke({"document_text": text})
        end_time = time.time()
        
        if session_logger:
            session_logger.log_llm_call(
                component="Document Validation",
                message=f"LLM validation for {category}",
                prompt=prompt_text,
                response=response.dict(),
                model="gpt-4o-mini",
                start_time=start_time,
                end_time=end_time,
                context={
                    "category": category,
                    "evidence": text[:100] + "..."
                }
            )

        return response.decision.lower() == "yes"

    except (ValueError, RuntimeError) as e:
        if session_logger:
            session_logger.log(
                component="Document Validation",
                message=f"LLM validation for {category}",
                decision="Rejected",
                reason=f"LLM Validation Error: {str(e)}",
                level="ERROR",
                context={"category": category}
            )
        return False