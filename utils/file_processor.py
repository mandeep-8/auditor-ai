#utils/file_processor.py
from utils.content_validator import extract_text_from_file, validate_document, process_org_chart_image
from datetime import datetime
import os

def process_file(file, category, file_path, session_logger):
    """Process a single file: save, extract text, and validate."""
    result = {"path": file_path, "text": None, "valid": False, "name": file.name, "error": None, "hierarchy": []}
    
    # Step 1: Save file
    session_logger.log("Document Validation", f"File={file.name}, Category={category}, Step=Saving")
    try:
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        session_logger.log("Document Validation", f"File={file.name}, Category={category}, Step=Saving", decision="Accepted", reason="File saved successfully")
    except (PermissionError, OSError) as e:
        result["error"] = f"Failed to save: {str(e)}"
        session_logger.log(
            "Document Validation",
            f"File={file.name}, Category={category}, Step=Saving",
            decision="Rejected",
            reason=f"Failed to save: {str(e)}"
        )
        return result

    # Step 2: Extract text and hierarchy for org charts, only text for others
    session_logger.log("Document Validation", f"File={file.name}, Category={category}, Step=Text Extraction")
    try:
        file_extension = os.path.splitext(file_path)[1].lower().lstrip(".")
        if category == "Organizational Chart" and file_extension in ["jpg", "jpeg", "png"]:
            text, hierarchy = process_org_chart_image(file_path, session_logger.session_id, session_logger)
            result["text"] = text
            result["hierarchy"] = hierarchy
        else:
            text = extract_text_from_file(file_path, session_logger)
            result["text"] = text
        session_logger.log(
            "Document Validation",
            f"File={file.name}, Category={category}, Step=Text Extraction",
            decision="Accepted",
            reason="Text extracted successfully",
            source=f"Text Preview={text[:100]}..." if text else None
        )
    except Exception as e:
        result["error"] = f"Failed to extract text: {str(e)}"
        session_logger.log(
            "Document Validation",
            f"File={file.name}, Category={category}, Step=Text Extraction",
            decision="Rejected",
            reason=f"Failed to extract text: {str(e)}"
        )
        return result

    # Step 3: Validate document
    session_logger.log("Document Validation", f"File={file.name}, Category={category}, Step=Validation")
    try:
        is_valid = validate_document(text, category, session_logger)
        result["valid"] = is_valid
        if not is_valid:
            result["error"] = "Document validation failed"
        session_logger.log(
            "Document Validation",
            f"File={file.name}, Category={category}, Step=Validation",
            decision="Accepted" if is_valid else "Rejected",
            reason="See LLM response for details"
        )
    except RuntimeError as e:
        result["error"] = f"Failed to validate: {str(e)}"
        session_logger.log(
            "Document Validation",
            f"File={file.name}, Category={category}, Step=Validation",
            decision="Rejected",
            reason=f"Failed to validate: {str(e)}"
        )
        return result
    except Exception as e:
        result["error"] = f"Unexpected validation error: {str(e)}"
        session_logger.log(
            "Document Validation",
            f"File={file.name}, Category={category}, Step=Validation",
            decision="Rejected",
            reason=f"Unexpected validation error: {str(e)}"
        )
        return result

    return result