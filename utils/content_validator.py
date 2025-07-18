#utils/content_validator.py
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os
import time
from dotenv import load_dotenv
from utils.session_logger import get_session_logger
from config import LOG_DIR
from functools import lru_cache
from PIL import Image
from io import BytesIO
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

load_dotenv()

class ValidationResponse(BaseModel):
    decision: str = Field(description="Either 'Yes' or 'No'")
    explanation: str = Field(description="Brief explanation of the decision (1-2 sentences)")
    confidence: int = Field(description="Confidence score from 0 to 100")

@lru_cache(maxsize=1000)
def cached_extract_text(file_path, file_type):
    """Cache extracted text from a file based on its type."""
    session_logger = get_session_logger(LOG_DIR, session_id=os.path.basename(file_path))  # Temporary logger for caching
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
        elif file_type in ["jpg", "png"]:
            # Initialize DocTR OCR predictor
            ocr = ocr_predictor(pretrained=True)
            # Load image
            doc = DocumentFile.from_images(file_path)
            # Perform OCR
            result = ocr(doc)
            # Extract text from all pages
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
        return text[:10000]  # Truncate to ~10,000 chars
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

def extract_text_from_file(file_path, session_logger=None):
    """Extract text from a file using appropriate LangChain loader or DocTR for images based on extension."""
    start_time = time.time()
    file_extension = os.path.splitext(file_path)[1].lower().lstrip(".")
    file_type = file_extension if file_extension in ["pdf", "docx", "xlsx", "jpg", "png"] else None
    
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