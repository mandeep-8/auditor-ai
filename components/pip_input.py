#components/pip_input.py
import streamlit as st
import os
import re
import shutil
import logging
import hashlib
from utils.validators import validate_pip_inputs
from utils.file_processor import process_file
from utils.session_logger import get_session_logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from config import UPLOAD_DIR, LOG_DIR, SESSIONS_DIR, OUTPUT_DIR
from pages.cdd_processing import process_cdd

def normalize_text(text):
    """Normalize text for comparison by removing extra spaces and converting to lowercase."""
    return re.sub(r"\s+", " ", text).strip().lower()

def get_file_identifier(file):
    """Generate a unique identifier for a file based on name and content length."""
    file.seek(0)
    content = file.read()
    file.seek(0)
    return hashlib.md5((file.name + str(len(content))).encode()).hexdigest()

def validate_uploaded_file(files, category, progress_container, status_container, session_logger):
    """Validate a list of uploaded files in parallel with per-file name, progress bar, and tick/cross."""
    if not files:
        session_logger.log(
            component="Document Validation",
            message=f"No files uploaded for {category}",
            decision="Skipped",
            reason="No files provided",
            level="INFO",
            context={"category": category}
        )
        return [], False

    results = []
    all_valid = True
    file_ui = {}

    with status_container:
        for file in files:
            with st.container():
                file_ui[file.name] = {
                    "name_display": st.markdown(f"{file.name}"),
                    "progress_text": st.text("0%"),
                    "progress_bar": st.progress(0),
                    "status_key": f"status_{file.name}_{category}",
                    "progress_key": f"progress_{file.name}_{category}"
                }
                st.session_state[file_ui[file.name]["status_key"]] = "0%"
                st.session_state[file_ui[file.name]["progress_key"]] = 0.0

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {
            executor.submit(
                process_file, 
                file, 
                category, 
                os.path.join(UPLOAD_DIR, file.name), 
                session_logger
            ): file for file in files
        }
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                file_id = get_file_identifier(file)
                result["file_id"] = file_id
                st.session_state[file_ui[file.name]["progress_key"]] = 1.0
                st.session_state[file_ui[file.name]["status_key"]] = "100%"
                file_ui[file.name]["progress_text"].text(st.session_state[file_ui[file.name]["status_key"]])
                file_ui[file.name]["progress_bar"].progress(st.session_state[file_ui[file.name]["progress_key"]])
                
                if result["valid"]:
                    file_ui[file.name]["name_display"].markdown(f"{file.name} ✅")
                else:
                    file_ui[file.name]["name_display"].markdown(f"{file.name} ❌ ({result['error']})")
                    all_valid = False
                
                results.append(result)
                session_logger.log(
                    component="Document Validation",
                    message=f"Validate File {file.name}",
                    decision="Accepted" if result["valid"] else "Rejected",
                    context={
                        "file": file.name,
                        "category": category,
                        "reason": result.get("error", "File validated successfully"),
                        "evidence": result["text"][:100] + "..." if result["text"] else None
                    },
                    level="INFO"
                )
            except Exception as e:
                file_id = get_file_identifier(file)
                st.session_state[file_ui[file.name]["progress_key"]] = 1.0
                st.session_state[file_ui[file.name]["status_key"]] = "100%"
                file_ui[file.name]["progress_text"].text(st.session_state[file_ui[file.name]["status_key"]])
                file_ui[file.name]["name_display"].markdown(f"{file.name} ❌ (Unexpected error: {str(e)})")
                all_valid = False
                results.append({"path": None, "text": None, "valid": False, "name": file.name, "file_id": file_id, "error": str(e), "hierarchy": []})
                
                session_logger.log(
                    component="Document Validation",
                    message=f"File={file.name}, Category={category}",
                    decision="Rejected",
                    reason=f"Unexpected error: {str(e)}",
                    level="ERROR",
                    context={"file": file.name, "category": category}
                )

    return results, all_valid

def show_pip_input_form():
    """Display the PIP input form with enhanced logging and navigation to cdd_processing."""
    st.subheader("📂 Enter Preliminary Information")
    
    session_logger = get_session_logger(LOG_DIR, st.session_state.session_id)
    
    if "org_chart_validation" not in st.session_state:
        st.session_state.org_chart_validation = {"results": [], "validated": False}
    if "governance_doc_validation" not in st.session_state:
        st.session_state.governance_doc_validation = {"results": [], "validated": False}
    
    with st.sidebar:
        log_level = st.selectbox(
            "Log Level",
            ["INFO", "DEBUG", "WARNING", "ERROR"],
            index=["INFO", "DEBUG", "WARNING", "ERROR"].index(os.getenv("LOG_LEVEL", "INFO").upper()),
            key="log_level"
        )
        if log_level != os.getenv("LOG_LEVEL", "INFO").upper():
            os.environ["LOG_LEVEL"] = log_level
            session_logger.log(
                component="UI Interaction",
                message="Log level changed",
                decision="Accepted",
                reason=f"Log level set to {log_level}",
                level="INFO",
                context={"log_level": log_level}
            )

    with st.form("pip_form"):
        control_owner_name = st.text_input("Control Owner Name*", "")
        control_owner_title = st.text_input("Control Owner Job Title & Role*", "")
        control_owner_email = st.text_input("Control Owner Email Address*", "")
        audit_period = st.text_input("Audit Period* (e.g., 01/01/2025 - 31/12/2025)", "")
        
        session_logger.log(
            component="PIP Input",
            message="User input provided",
            level="DEBUG",
            context={
                "user_input": {
                    "control_owner_name": control_owner_name,
                    "control_owner_title": control_owner_title,
                    "control_owner_email": control_owner_email,
                    "audit_period": audit_period
                }
            }
        )

        org_chart_container = st.container()
        with org_chart_container:
            org_charts = st.file_uploader("Organisational Chart (PDF, DOCX, XLSX, JPG, PNG)", type=["pdf", "docx", "xlsx", "jpg", "png"], key="org_chart", accept_multiple_files=True)
            org_chart_progress = st.container()
            org_chart_status = st.container()
        org_chart_missing = st.toggle("I don't have the Organisational Chart")
        
        governance_doc_container = st.container()
        with governance_doc_container:
            governance_docs = st.file_uploader("Governance Documentation (PDF, DOCX, XLSX, JPG, PNG)", type=["pdf", "docx", "xlsx", "jpg", "png"], key="governance_doc", accept_multiple_files=True)
            governance_progress = st.container()
            governance_status = st.container()
        governance_doc_missing = st.toggle("I don't have the Governance Document")
        
        current_org_chart_ids = [get_file_identifier(f) for f in org_charts] if org_charts else []
        current_gov_doc_ids = [get_file_identifier(f) for f in governance_docs] if governance_docs else []
        
        st.session_state.org_chart_validation["results"] = [
            r for r in st.session_state.org_chart_validation["results"]
            if r["file_id"] in current_org_chart_ids
        ]
        st.session_state.governance_doc_validation["results"] = [
            r for r in st.session_state.governance_doc_validation["results"]
            if r["file_id"] in current_gov_doc_ids
        ]
        
        session_logger.log(
            component="PIP Input",
            message="Toggle states updated",
            level="DEBUG",
            context={
                "org_chart_missing": org_chart_missing,
                "governance_doc_missing": governance_doc_missing
            }
        )

        if org_charts:
            new_org_charts = [
                f for f in org_charts
                if get_file_identifier(f) not in [r["file_id"] for r in st.session_state.org_chart_validation["results"]]
            ]
            if new_org_charts:
                new_results, new_all_valid = validate_uploaded_file(
                    new_org_charts, "Organizational Chart", org_chart_progress, org_chart_status, session_logger
                )
                st.session_state.org_chart_validation["results"].extend(new_results)
                st.session_state.org_chart_validation["validated"] = True
                session_logger.log(
                    component="PIP Validation",
                    message=f"Organizational Charts Validated: {new_all_valid}",
                    decision="Accepted" if new_all_valid else "Rejected",
                    reason=f"Files: {[result['name'] for result in new_results]}",
                    level="INFO",
                    context={"files": [result["name"] for result in new_results]}
                )
        elif not org_charts and not st.session_state.org_chart_validation["validated"]:
            st.session_state.org_chart_validation = {"results": [], "validated": False}

        if governance_docs:
            new_gov_docs = [
                f for f in governance_docs
                if get_file_identifier(f) not in [r["file_id"] for r in st.session_state.governance_doc_validation["results"]]
            ]
            if new_gov_docs:
                new_results, new_all_valid = validate_uploaded_file(
                    new_gov_docs, "Governance Document", governance_progress, governance_status, session_logger
                )
                st.session_state.governance_doc_validation["results"].extend(new_results)
                st.session_state.governance_doc_validation["validated"] = True
                session_logger.log(
                    component="PIP Validation",
                    message=f"Governance Documents Validated: {new_all_valid}",
                    decision="Accepted" if new_all_valid else "Rejected",
                    reason=f"Files: {[result['name'] for result in new_results]}",
                    level="INFO",
                    context={"files": [result["name"] for result in new_results]}
                )
        elif not governance_docs and not st.session_state.governance_doc_validation["validated"]:
            st.session_state.governance_doc_validation = {"results": [], "validated": False}

        submitted = st.form_submit_button("Validate")
        if submitted:
            session_logger.log(
                component="UI Interaction",
                message="Validate button clicked",
                decision="Accepted",
                level="INFO",
                context={}
            )
            
            valid, message = validate_pip_inputs(
                control_owner_name, control_owner_title, control_owner_email, org_charts, audit_period
            )
            session_logger.log(
                component="PIP Validation",
                message="Text Inputs Validation",
                decision="Accepted" if valid else "Rejected",
                reason=message,
                level="INFO",
                context={"user_input": {
                    "control_owner_name": control_owner_name,
                    "control_owner_title": control_owner_title,
                    "control_owner_email": control_owner_email,
                    "audit_period": audit_period
                }}
            )
            
            if not valid:
                st.error(message)
                return None

            if not org_chart_missing and not st.session_state.org_chart_validation["results"]:
                st.error("⚠️ Please upload at least one Organisational Chart or mark it as missing.")
                session_logger.log(
                    component="PIP Validation",
                    message="Submission Check",
                    decision="Rejected",
                    reason="No Organizational Chart uploaded and not marked as missing",
                    level="ERROR",
                    context={}
                )
                return None
            if not governance_doc_missing and not st.session_state.governance_doc_validation["results"]:
                st.error("⚠️ Please upload at least one Governance Document or mark it as missing.")
                session_logger.log(
                    component="PIP Validation",
                    message="Submission Check",
                    decision="Rejected",
                    reason="No Governance Document uploaded and not marked as missing",
                    level="ERROR",
                    context={}
                )
                return None

    text_valid, message = validate_pip_inputs(control_owner_name, control_owner_title, control_owner_email, org_charts, audit_period)
    org_chart_valid = org_chart_missing or (st.session_state.org_chart_validation["results"] and all(result["valid"] for result in st.session_state.org_chart_validation["results"]))
    governance_valid = governance_doc_missing or (st.session_state.governance_doc_validation["results"] and all(result["valid"] for result in st.session_state.governance_doc_validation["results"]))
    show_submit = text_valid and org_chart_valid and governance_valid

    if show_submit:
        if st.button("SUBMIT"):
            session_logger.log(
                component="UI Interaction",
                message="SUBMIT button clicked",
                decision="Accepted",
                level="INFO",
                context={}
            )
            
            session_dir = os.path.join(SESSIONS_DIR, st.session_state.session_id)
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir)
                session_logger.log(
                    component="PIP Submission",
                    message=f"Cleaned up vector store for session {st.session_state.session_id}",
                    decision="Accepted",
                    reason="Ensuring fresh vector store for new session",
                    level="DEBUG",
                    context={"session_dir": session_dir}
                )
            
            st.session_state.pip_control_owner_name = control_owner_name
            st.session_state.pip_control_owner_title = control_owner_title
            st.session_state.pip_audit_period = audit_period
            pip_info_path = os.path.join(OUTPUT_DIR, "pip_info.txt")
            with open(pip_info_path, "w") as info_file:
                valid_org_charts = [result["name"] for result in st.session_state.org_chart_validation["results"] if result["valid"]]
                valid_governance_docs = [result["name"] for result in st.session_state.governance_doc_validation["results"] if result["valid"]]
                info_file.write(f'Audit Period: {audit_period}\n')
                if valid_org_charts:
                    info_file.write(f'Organisational Chart Names: {", ".join(valid_org_charts)}\n')
                if valid_governance_docs:
                    info_file.write(f'Governance Document Names: {", ".join(valid_governance_docs)}\n')
            session_logger.log(
                component="PIP Submission",
                message=f"Valid documents and audit period written to pip_info.txt",
                decision="Accepted",
                reason=f"Org Charts={valid_org_charts}, Governance Docs={valid_governance_docs}, Audit Period={audit_period}",
                level="INFO",
                context={"org_charts": valid_org_charts, "governance_docs": valid_governance_docs, "audit_period": audit_period}
            )

            pip_conclusion_path = os.path.join(OUTPUT_DIR, "pip_conclusion.txt")
            with open(pip_conclusion_path, "w") as conclusion_file:
                conclusion_file.write("")
            session_logger.log(
                component="PIP Submission",
                message="Cleared pip_conclusion.txt",
                decision="Accepted",
                level="INFO",
                context={}
            )

            if governance_doc_missing:
                with open(pip_conclusion_path, "a") as conclusion_file:
                    conclusion_file.write("Governance documentation missing\n")
                st.warning("⚠️ Governance Document marked as not available.")
                session_logger.log(
                    component="PIP Submission",
                    message="Governance Document",
                    decision="Skipped",
                    reason="Marked as missing",
                    level="WARNING",
                    context={}
                )

            if not st.session_state.org_chart_validation["results"] and org_chart_missing:
                with open(pip_conclusion_path, "a") as conclusion_file:
                    conclusion_file.write("Organisational Chart missing\n")
                st.warning("⚠️ Organisational Chart invalidated.")
                session_logger.log(
                    component="PIP Submission",
                    message="Organizational Chart",
                    decision="Invalidated",
                    reason="Marked as missing",
                    level="WARNING",
                    context={}
                )
            else:
                name_found = False
                title_matched = False
                matched_file = None
                matched_index = -1

                normalized_name = normalize_text(control_owner_name)
                normalized_title = normalize_text(control_owner_title)

                for result in st.session_state.org_chart_validation["results"]:
                    if not result["valid"]:
                        continue
                    org_chart_text = result["text"]
                    lines = org_chart_text.split('\n')
                    normalized_lines = [normalize_text(line) for line in lines]

                    for idx, line in enumerate(normalized_lines):
                        if normalized_name in line:
                            name_found = True
                            matched_index = idx
                            matched_file = result["name"]
                            break
                    if name_found:
                        start = max(matched_index - 3, 0)
                        end = min(matched_index + 4, len(normalized_lines))
                        for i in range(start, end):
                            if normalized_title in normalized_lines[i]:
                                title_matched = True
                                break
                    if name_found and title_matched:
                        break

                if name_found:
                    session_logger.log(
                        component="PIP Validation",
                        message=f"Name match found in {matched_file} at line {matched_index + 1}",
                        decision="Accepted",
                        reason=f"Name '{control_owner_name}' found in org chart",
                        level="INFO",
                        context={"file": matched_file, "line_number": matched_index + 1}
                    )
                else:
                    with open(pip_conclusion_path, "a") as conclusion_file:
                        conclusion_file.write(f"Could not find control owner name ({control_owner_name}) in any Organisational Chart\n")
                    st.warning(f"⚠️ Could not find {control_owner_name} in any Organisational Chart.")
                    session_logger.log(
                        component="PIP Validation",
                        message=f"Name match for {control_owner_name}",
                        decision="Rejected",
                        reason="Control owner name not found in org chart",
                        level="WARNING",
                        context={}
                    )

                if title_matched:
                    session_logger.log(
                        component="PIP Validation",
                        message=f"Title match found in {matched_file} at line {matched_index + 1}",
                        decision="Accepted",
                        reason=f"Title '{control_owner_title}' matched in org chart",
                        level="INFO",
                        context={"file": matched_file, "line_number": matched_index + 1}
                    )
                elif name_found:
                    with open(pip_conclusion_path, "a") as conclusion_file:
                        conclusion_file.write(f"Could not match control owner title ({control_owner_title}) in any Organisational Chart\n")
                    st.warning(f"⚠️ Could not match {control_owner_title} in any Organisational Chart.")
                    session_logger.log(
                        component="PIP Validation",
                        message=f"Title match for {control_owner_title}",
                        decision="Rejected",
                        reason="Control owner title not matched in org chart",
                        level="WARNING",
                        context={}
                    )

            st.session_state.pip_submitted = True
            session_logger.log(
                component="PIP Submission",
                message="PIP form submitted successfully",
                decision="Accepted",
                reason="All validations passed, navigating to CDD processing",
                level="INFO",
                context={}
            )
            
            try:
                st.switch_page("pages/cdd_processing.py")
            except Exception as e:
                st.error(f"Failed to navigate to CDD Processing page: {str(e)}")
                session_logger.log(
                    component="PIP Submission",
                    message="Navigation to CDD Processing",
                    decision="Rejected",
                    reason=f"Navigation error: {str(e)}",
                    level="ERROR",
                    context={}
                )