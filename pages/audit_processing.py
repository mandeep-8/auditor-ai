#pages/audit_processing.py
import streamlit as st
import pandas as pd
import os
import json
import re
from utils.question_answering import get_qa_chain
from utils.session_logger import get_session_logger
from utils.content_validator import extract_text_from_file
from config import INPUT_DIR, LOG_DIR, SESSIONS_DIR, OUTPUT_DIR
from functools import lru_cache
from datetime import datetime

# Cache QA chain results to avoid redundant calls
@lru_cache(maxsize=1000)
def cached_qa_invoke(query, context):
    """Cache QA chain results to improve performance."""
    qa_chain = st.session_state.qa_chain
    return qa_chain.invoke({"query": query, "context": context})["result"].strip()

# SOD conflict detection data
privilege_aliases = {
    "Granting/Modifying User Access": 1,
    "Authorizing User Access": 2,
    "Authorizing System Access": 3,
    "Application Modification": 4,
    "System Modification": 5,
    "Move Application Changes into Production": 6,
    "Authorization of Production Modification": 7,
    "Data Entry": 8,
    "System Administration": 9,
    "Quality Assurance": 10,
    "Network Security": 11,
    "Network Administration": 12,
    "Application Administration": 13
}

privilege_id_to_name = {v: k for k, v in privilege_aliases.items()}

controller = [
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]
]

def check_all_user_conflicts_from_tickets(ticket_rows, controller, privilege_aliases):
    """Check for SOD conflicts in user privileges."""
    results = []
    id_to_name = {v: k for k, v in privilege_aliases.items()}

    for row in ticket_rows:
        user_token = row['User ID'].strip()
        privileges_raw = row.get('Privileges', '')
        privilege_texts = [p.strip() for p in re.split('[,;]', privileges_raw) if p.strip()] if privileges_raw else []

        privilege_ids = []
        unknown_privileges = []
        for priv in privilege_texts:
            pid = privilege_aliases.get(priv)
            if pid:
                privilege_ids.append(pid)
            else:
                unknown_privileges.append(priv)

        conflicts = []
        for i in range(len(privilege_ids)):
            for j in range(i + 1, len(privilege_ids)):
                p1, p2 = privilege_ids[i], privilege_ids[j]
                if controller[p1 - 1][p2 - 1] == 1:
                    conflicts.append({
                        "privilege_1_id": p1,
                        "privilege_1_name": id_to_name[p1],
                        "privilege_2_id": p2,
                        "privilege_2_name": id_to_name[p2]
                    })

        results.append({
            "user_token": user_token,
            "conflict_exists": len(conflicts) > 0,
            "conflicting_privileges": conflicts,
            "unknown_privileges": unknown_privileges
        })

    return results

def normalize_text(text):
    """Normalize text for exact comparison."""
    return str(text).strip().lower()

def validate_approver_title(row, cdd_df, question_col, answer_col, org_chart_text, session_logger):
    """Validate approver title using CDD, line manager, and organizational chart."""
    approver_name = row.get("Approver Name", None) if pd.notna(row.get("Approver Name", None)) else None
    approver_title = row.get("Approver Title", None) if pd.notna(row.get("Approver Title", None)) else None
    line_manager = row.get("Line Manager", None) if pd.notna(row.get("Line Manager", None)) else None
    user_id = row.get("User ID", "Unknown")

    if not approver_name or not approver_title:
        return False, f"Validation failed for User ID {user_id}: Approver Name='{approver_name}', Approver Title='{approver_title}' (missing required fields)"

    # Step 1: Check if approver title matches CDD answer
    if st.session_state.get("qa_chain") is not None:
        for _, cdd_row in cdd_df.iterrows():
            answer = cdd_row[answer_col] if answer_col in cdd_row and pd.notna(cdd_row[answer_col]) else ""
            if not answer:
                continue
            try:
                query = f"What is the job title of {approver_name} in this answer?"
                cdd_title = cached_qa_invoke(query, answer)
                if cdd_title and cdd_title.lower() not in ["unknown", "not found", "none"] and cdd_title.lower() == approver_title.lower():
                    session_logger.log(
                        "Audit Processing",
                        f"Approver Title Validation: User ID={user_id}",
                        decision="Accepted",
                        reason=f"Approver {approver_name}'s title {approver_title} matches CDD",
                        source=f"CDD Answer={answer[:50]}..."
                    )
                    return True, f"Validation passed for User ID {user_id}: Approver Name='{approver_name}', Approver Title='{approver_title}' matches CDD title='{cdd_title}'"
            except Exception as e:
                session_logger.log(
                    "Audit Processing",
                    f"Approver Title Validation: User ID={user_id}",
                    decision="Error",
                    reason=f"CDD title check error: {str(e)}",
                    source=f"Approver={approver_name}"
                )

    # Step 2: Check if approver name matches line manager
    if line_manager and normalize_text(approver_name) == normalize_text(line_manager):
        session_logger.log(
            "Audit Processing",
            f"Approver Title Validation: User ID={user_id}",
            decision="Accepted",
            reason=f"Approver {approver_name} matches line manager",
            source=f"Line Manager={line_manager}"
        )
        return True, f"Validation passed for User ID {user_id}: Approver Name='{approver_name}' matches Line Manager='{line_manager}'"

    # Step 3: Check if approver title is managerial in organizational chart
    if org_chart_text:
        try:
            query = f"""
You are an expert in organizational structure and corporate hierarchy tasked with determining whether a given job title or designation is considered managerial.

A managerial role meets any of the following conditions:
- Direct or indirect authority over employees (e.g., supervises a team or department).
- Responsibility for project, resource, or budget management.
- Involvement in strategic decision-making, planning, or cross-functional leadership.
- Appears in the organizational chart above individual contributors and positioned with people-reporting lines.

To assess the managerial nature of the designation, evaluate the following:
1. Does this role appear above others in the organizational chart?
2. Does this role supervise or coordinate others?
3. Does this role make strategic or operational decisions?
4. Does this role influence performance reviews, hiring, or resource allocation?

Use job title semantics (e.g., Manager, Director, VP), org chart position, and responsibility scope to guide your reasoning.

Now evaluate the following designation:
{approver_title}

Respond with a JSON object containing:
- "Managerial": "Yes" or "No"
- "Reasoning": Provide a clear explanation referring to hierarchy, scope, and responsibilities.
"""
            result = cached_qa_invoke(query, org_chart_text) if st.session_state.get("qa_chain") is not None else ""
            if not result:
                session_logger.log(
                    "Audit Processing",
                    f"Approver Title Validation: User ID={user_id}",
                    decision="Error",
                    reason=f"Empty LLM response for Approver Title='{approver_title}'",
                    source="Org Chart Query"
                )
                return False, f"Validation failed for User ID {user_id}: Empty LLM response for Approver Title='{approver_title}'"
            try:
                # Preprocess the response to extract valid JSON
                json_match = re.search(r'\{.*?\}', result, re.DOTALL)
                if not json_match:
                    session_logger.log(
                        "Audit Processing",
                        f"Approver Title Validation: User ID={user_id}",
                        decision="Error",
                        reason="No valid JSON found in LLM response",
                        source=f"Raw Response={result[:100]}..."
                    )
                    return False, f"Validation failed for User ID {user_id}: No valid JSON found in LLM response for Approver Title='{approver_title}'"
                
                json_str = json_match.group(0)
                response = json.loads(json_str)
                is_managerial = response.get("Managerial", "No").lower() == "yes"
                reasoning = response.get("Reasoning", "No reasoning provided")
                if is_managerial:
                    session_logger.log(
                        "Audit Processing",
                        f"Approver Title Validation: User ID={user_id}",
                        decision="Accepted",
                        reason=f"Approver title {approver_title} is managerial: {reasoning}",
                        source=f"Org Chart Query, Raw Response={result[:100]}..."
                    )
                    return True, f"Validation passed for User ID {user_id}: Approver Title='{approver_title}' is managerial: {reasoning}"
                else:
                    session_logger.log(
                        "Audit Processing",
                        f"Approver Title Validation: User ID={user_id}",
                        decision="Rejected",
                        reason=f"Approver title {approver_title} not managerial: {reasoning}",
                        source=f"Org Chart Query, Raw Response={result[:100]}..."
                    )
                    return False, f"Validation failed for User ID {user_id}: Approver Title='{approver_title}' not managerial: {reasoning}"
            except json.JSONDecodeError as e:
                session_logger.log(
                    "Audit Processing",
                    f"Approver Title Validation: User ID={user_id}",
                    decision="Error",
                    reason=f"Failed to parse LLM response: {str(e)}",
                    source=f"Raw Response={result[:100]}..."
                )
                return False, f"Validation failed for User ID {user_id}: Error parsing LLM response for Approver Title='{approver_title}': {str(e)}"
        except Exception as e:
            session_logger.log(
                "Audit Processing",
                f"Approver Title Validation: User ID={user_id}",
                decision="Error",
                reason=f"Org chart title check error: {str(e)}",
                source=f"Approver={approver_name}, Raw Response={result[:100] if 'result' in locals() else 'N/A'}..."
            )
            return False, f"Validation failed for User ID {user_id}: Error checking Approver Title='{approver_title}' in organizational chart: {str(e)}"
    else:
        session_logger.log(
            "Audit Processing",
            f"Approver Title Validation: User ID={user_id}",
            decision="Accepted",
            reason="No org chart available, assuming valid approver title",
            source=f"Approver={approver_name}"
        )
        return True, f"Validation passed for User ID {user_id}: No org chart available, assuming Approver Name='{approver_name}', Approver Title='{approver_title}' is valid"

def map_column_names(df, session_logger):
    """Map actual column names to expected column names."""
    column_mapping = {
        "user id": "User ID",
        "approver name": "Approver Name",
        "approver job title": "Approver Title",
        "approver title": "Approver Title",
        "line manager": "Line Manager",
        "requested role(s)": "Access Requested",
        "requested role": "Access Requested",
        "granted role(s)": "Access Granted",
        "granted role": "Access Granted",
        "user name": "User Name",
        "provisioner name": "Granter",
        "privileges": "Privileges",
        "privileges granted": "Privileges",
        "sod check performed by mgmt": "SOD Check Performed by Mgmt",
        "date of access created": "Date of Access Created",
        "date of last modified": "Date of Last Modified",
        "access approval date": "Access Approval Date"
    }
    
    rename_dict = {}
    for actual_col in df.columns:
        normalized_col = actual_col.lower().replace(" ", "")
        for expected_col, mapped_col in column_mapping.items():
            if normalized_col == expected_col.lower().replace(" ", ""):
                rename_dict[actual_col] = mapped_col
                break
    
    df = df.rename(columns=rename_dict)
    
    session_logger.log(
        "Audit Processing",
        "Column Mapping",
        decision="Accepted",
        reason=f"Mapped columns: {rename_dict}",
        source=f"Original columns: {list(df.columns)}"
    )
    
    return df

def classify_user(row, audit_period_start, audit_period_end, session_logger):
    """Classify user as new, modified, both, or neither based on audit period."""
    user_id = row.get("User ID", "Unknown")
    created_date = row.get("Date of Access Created", None)
    modified_date = row.get("Date of Last Modified", None)
    
    is_new_user = False
    is_modified_user = False
    reason = []

    try:
        if created_date and pd.notna(created_date):
            try:
                created_date = pd.to_datetime(created_date, errors='coerce')
                if pd.notna(created_date) and audit_period_start <= created_date <= audit_period_end:
                    is_new_user = True
                    reason.append(f"Access created on {created_date.strftime('%d/%m/%Y')} is within audit period ({audit_period_start.strftime('%d/%m/%Y')} - {audit_period_end.strftime('%d/%m/%Y')}).")
                else:
                    reason.append(f"Access created on {created_date.strftime('%d/%m/%Y')} is outside audit period ({audit_period_start.strftime('%d/%m/%Y')} - {audit_period_end.strftime('%d/%m/%Y')}).")
            except Exception as e:
                reason.append(f"Invalid Date of Access Created: {str(e)}")
        else:
            reason.append("Date of Access Created is missing.")

        if modified_date and pd.notna(modified_date):
            try:
                modified_date = pd.to_datetime(modified_date, errors='coerce')
                if pd.notna(modified_date) and audit_period_start <= modified_date <= audit_period_end:
                    is_modified_user = True
                    reason.append(f"Access modified on {modified_date.strftime('%d/%m/%Y')} is within audit period ({audit_period_start.strftime('%d/%m/%Y')} - {audit_period_end.strftime('%d/%m/%Y')}).")
                else:
                    reason.append(f"Access modified on {modified_date.strftime('%d/%m/%Y')} is outside audit period ({audit_period_start.strftime('%d/%m/%Y')} - {audit_period_end.strftime('%d/%m/%Y')}).")
            except Exception as e:
                reason.append(f"Invalid Date of Last Modified: {str(e)}")
        else:
            reason.append("Date of Last Modified is missing or not applicable.")

        if is_new_user and is_modified_user:
            status = "New User and Modified User"
        elif is_new_user:
            status = "New User"
        elif is_modified_user:
            status = "Modified User"
        else:
            status = "Neither New nor Modified User"

        session_logger.log(
            "Audit Processing",
            f"User Classification: User ID={user_id}",
            decision="Accepted",
            reason=f"Classified as {status}: {'; '.join(reason)}",
            source=f"Created={created_date}, Modified={modified_date}"
        )

        return {
            "is_new_user": is_new_user,
            "is_modified_user": is_modified_user,
            "status": status,
            "reason": "; ".join(reason)
        }

    except Exception as e:
        session_logger.log(
            "Audit Processing",
            f"User Classification: User ID={user_id}",
            decision="Error",
            reason=f"Classification error: {str(e)}",
            source=f"Created={created_date}, Modified={modified_date}"
        )
        return {
            "is_new_user": False,
            "is_modified_user": False,
            "status": "Error",
            "reason": f"Classification error: {str(e)}"
        }

def process_audit():
    """Perform control tests and display results with approver title validation and user classification."""
    session_logger = get_session_logger(LOG_DIR, st.session_state.session_id)
    st.subheader("🔍 Audit Processing")
    session_logger.log("Audit Processing", "Started audit processing")

    # Check required session state variables
    if not st.session_state.get("cdd_saved", False) or not st.session_state.get("cdd_validated", False):
        st.error("⚠️ CDD not saved or validated. Please process and save the CDD first.")
        session_logger.log(
            "Audit Processing",
            "CDD Check",
            decision="Rejected",
            reason="CDD not saved or validated"
        )
        return

    if "excel_df" not in st.session_state or st.session_state.excel_df.empty:
        st.error("⚠️ CDD data not initialized. Please process the CDD first.")
        session_logger.log(
            "Audit Processing",
            "Excel Data Check",
            decision="Rejected",
            reason="CDD Excel data not initialized"
        )
        return

    qa_chain = st.session_state.get("qa_chain", None)
    cdd_text = st.session_state.get("cdd_text", "")

    # Load audit period
    audit_period = st.session_state.get("pip_audit_period", "")
    if not audit_period:
        st.error("⚠️ Audit period not specified. Please submit the PIP form with audit period.")
        session_logger.log(
            "Audit Processing",
            "Audit Period Check",
            decision="Rejected",
            reason="Audit period not found in session state"
        )
        return

    try:
        start_date_str, end_date_str = audit_period.split(" - ")
        audit_period_start = pd.to_datetime(start_date_str, dayfirst=True)
        audit_period_end = pd.to_datetime(end_date_str, dayfirst=True)
        if audit_period_start > audit_period_end:
            st.error("⚠️ Audit period start date must be before end date.")
            session_logger.log(
                "Audit Processing",
                "Audit Period Validation",
                decision="Rejected",
                reason=f"Invalid audit period: Start date {start_date_str} is after end date {end_date_str}"
            )
            return
        session_logger.log(
            "Audit Processing",
            "Audit Period Validation",
            decision="Accepted",
            reason=f"Audit period set: {start_date_str} - {end_date_str}"
        )
    except Exception as e:
        st.error(f"⚠️ Invalid audit period format. Use DD/MM/YYYY - DD/MM/YYYY. Error: {str(e)}")
        session_logger.log(
            "Audit Processing",
            "Audit Period Validation",
            decision="Rejected",
            reason=f"Invalid audit period format: {str(e)}"
        )
        return

    # Load updated CDD if available
    session_dir = os.path.join(SESSIONS_DIR, st.session_state.session_id)
    updated_excel_path = os.path.join(session_dir, "Control Design Document_updated.xlsx")
    if os.path.exists(updated_excel_path):
        try:
            cdd_df = pd.read_excel(updated_excel_path, engine="openpyxl", dtype=str)
            session_logger.log(
                "Audit Processing",
                "CDD Excel Load",
                decision="Accepted",
                reason="Updated CDD loaded successfully",
                source=f"File={updated_excel_path}"
            )
        except Exception as e:
            session_logger.log(
                "Audit Processing",
                "CDD Excel Load",
                decision="Rejected",
                reason=f"Failed to load updated CDD: {str(e)}"
            )

    # Identify CDD columns
    actual_columns = {col.lower().replace(" ", ""): col for col in cdd_df.columns}
    question_col = next((col for key, col in actual_columns.items() if "controldesignunderstandingquestions" in key), None)
    answer_col = next((col for key, col in actual_columns.items() if "answers" in key), None)
    if not question_col or not answer_col:
        st.error(f"⚠️ CDD Excel must contain 'Control design understanding questions' and 'Answers' columns. Found: {list(cdd_df.columns)}")
        session_logger.log(
            "Audit Processing",
            "CDD Column Check",
            decision="Rejected",
            reason=f"Required columns not found: {list(cdd_df.columns)}"
        )
        return

    # Check CDD question 26 for SOD performed by management
    sod_by_management = False
    try:
        last_row = cdd_df.iloc[-1]
        if pd.notna(last_row[question_col]) and "SOD" in last_row[question_col].lower() and pd.notna(last_row[answer_col]):
            sod_by_management = normalize_text(last_row[answer_col]) == "yes"
            session_logger.log(
                "Audit Processing",
                "SOD Question Check",
                decision="Accepted",
                reason=f"SOD performed by management: {sod_by_management}",
                source=f"Question={last_row[question_col][:50]}..., Answer={last_row[answer_col]}"
            )
    except Exception as e:
        session_logger.log(
            "Audit Processing",
            "SOD Question Check",
            decision="Error",
            reason=f"Failed to check SOD question: {str(e)}",
            source="CDD last question"
        )

    # Load organizational chart text from pip_info.txt
    org_chart_text = ""
    pip_info_path = os.path.join(OUTPUT_DIR, "pip_info.txt")
    org_chart_filenames = []
    if os.path.exists(pip_info_path):
        try:
            with open(pip_info_path, "r") as info_file:
                for line in info_file:
                    if line.startswith("Organisational Chart Names:"):
                        org_chart_filenames = [name.strip() for name in line.split(":")[1].split(",") if name.strip()]
                        break
            session_logger.log(
                "Audit Processing",
                "Org Chart File Load",
                decision="Accepted",
                reason=f"Read org chart filenames from pip_info.txt: {org_chart_filenames}",
                source=f"File={pip_info_path}"
            )
        except Exception as e:
            session_logger.log(
                "Audit Processing",
                "Org Chart File Load",
                decision="Rejected",
                reason=f"Failed to read pip_info.txt: {str(e)}",
                source=f"File={pip_info_path}"
            )

    # Try loading each org chart file from uploads, session directory, then INPUT_DIR
    for org_chart_filename in org_chart_filenames:
        # Check uploads directory
        org_chart_path = os.path.join("Uploads", org_chart_filename)
        if os.path.exists(org_chart_path):
            try:
                org_chart_text = extract_text_from_file(org_chart_path, session_logger)
                if org_chart_text:
                    session_logger.log(
                        "Audit Processing",
                        "Org Chart Load",
                        decision="Accepted",
                        reason="Organizational chart text extracted",
                        source=f"File={org_chart_path}"
                    )
                    break
            except Exception as e:
                st.warning(f"⚠️ Failed to load organizational chart {org_chart_path}: {str(e)}")
                session_logger.log(
                    "Audit Processing",
                    "Org Chart Load",
                    decision="Rejected",
                    reason=f"Failed to load org chart {org_chart_path}: {str(e)}"
                )
        # Check session directory
        org_chart_path = os.path.join(session_dir, org_chart_filename)
        if os.path.exists(org_chart_path):
            try:
                org_chart_text = extract_text_from_file(org_chart_path, session_logger)
                if org_chart_text:
                    session_logger.log(
                        "Audit Processing",
                        "Org Chart Load",
                        decision="Accepted",
                        reason="Organizational chart text extracted",
                        source=f"File={org_chart_path}"
                    )
                    break
            except Exception as e:
                st.warning(f"⚠️ Failed to load organizational chart {org_chart_path}: {str(e)}")
                session_logger.log(
                    "Audit Processing",
                    "Org Chart Load",
                    decision="Rejected",
                    reason=f"Failed to load org chart {org_chart_path}: {str(e)}"
                )
        # Fallback to INPUT_DIR
        org_chart_path = os.path.join(INPUT_DIR, org_chart_filename)
        if os.path.exists(org_chart_path):
            try:
                org_chart_text = extract_text_from_file(org_chart_path, session_logger)
                if org_chart_text:
                    session_logger.log(
                        "Audit Processing",
                        "Org Chart Load",
                        decision="Accepted",
                        reason="Organizational chart text extracted",
                        source=f"File={org_chart_path}"
                    )
                    break
            except Exception as e:
                st.warning(f"⚠️ Failed to load organizational chart {org_chart_path}: {str(e)}")
                session_logger.log(
                    "Audit Processing",
                    "Org Chart Load",
                    decision="Rejected",
                    reason=f"Failed to load org chart {org_chart_path}: {str(e)}"
                )
    
    if not org_chart_text:
        st.warning("⚠️ No valid organizational chart found. Approver validation will skip org chart check.")
        session_logger.log(
            "Audit Processing",
            "Org Chart Load",
            decision="Rejected",
            reason="No valid organizational chart found"
        )

    # File uploader for testing data
    uploaded_file = st.file_uploader("Upload testing_data.xlsx", type=["xlsx"], key="testing_data_uploader")
    if not uploaded_file:
        session_logger.log(
            "Audit Processing",
            "Testing Data Upload",
            decision="Pending",
            reason="Waiting for user to upload testing data file"
        )
        return
    try:
        test_df = pd.read_excel(uploaded_file, engine="openpyxl", dtype=str)
        session_logger.log(
            "Audit Processing",
            "Testing Data Upload",
            decision="Accepted",
            reason="Testing data uploaded successfully",
            source=f"File={uploaded_file.name}"
        )
    except Exception as e:
        st.error(f"⚠️ Failed to read uploaded testing data file: {str(e)}")
        session_logger.log(
            "Audit Processing",
            "Testing Data Upload",
            decision="Rejected",
            reason=f"Failed to read uploaded testing data: {str(e)}"
        )
        return

    # Map column names to expected names
    test_df = map_column_names(test_df, session_logger)

    # Validate required columns in test data
    required_columns = [
        "User ID", "Approver Name", "Approver Title", "Line Manager", 
        "Access Requested", "Access Granted", "User Name", "Granter", 
        "Privileges", "Date of Access Created", "Date of Last Modified", 
        "Access Approval Date"
    ]
    missing_columns = [col for col in required_columns if col not in test_df.columns]
    if missing_columns:
        st.error(f"⚠️ Testing data Excel must contain columns: {', '.join(required_columns)}. Missing: {', '.join(missing_columns)}. Found: {list(test_df.columns)}")
        session_logger.log(
            "Audit Processing",
            "Testing Data Column Check",
            decision="Rejected",
            reason=f"Missing required columns: {missing_columns}"
        )
        return

    # Define control tests
    control_tests = [
        {"name": "New user access approved", "type": "Access approval", "applies_to": "new"},
        {"name": "Modified user access approved", "type": "Access approval", "applies_to": "modified"},
        {"name": "New user access consistent", "type": "Access requested = Access granted", "applies_to": "new"},
        {"name": "Modified user access consistent", "type": "Access requested = Access granted", "applies_to": "modified"},
        {"name": "New user access appropriateness", "type": "Access appropriateness", "applies_to": "new"},
        {"name": "Modified user access appropriateness", "type": "Access appropriateness", "applies_to": "modified"},
        {"name": "SoD: New user access", "type": "Approver = Granter/Creator/Provisioner", "applies_to": "new"},
        {"name": "SoD: Modified user access", "type": "Approver = Granter/Creator/Provisioner", "applies_to": "modified"},
        {"name": "Access approval date validation", "type": "Date validation", "applies_to": "both"}
    ]

    # Process rows and control tests in a single loop
    st.write("### Audit Results")
    results = []
    missing_titles = []
    user_classifications = []
    for idx, row in test_df.iterrows():
        user_id = row["User ID"]
        access_requested = row.get("Access Requested", "")
        access_granted = row.get("Access Granted", "")
        user_name = row.get("User Name", "")
        approver_name = row.get("Approver Name", "")
        granter = row.get("Granter", "")
        privileges = row.get("Privileges", "")

        # Classify user
        classification = classify_user(row, audit_period_start, audit_period_end, session_logger)
        user_classifications.append({
            "User ID": user_id,
            "Status": classification["status"],
            "Reason": classification["reason"]
        })

        # Validate approver title once per row
        is_valid_approver, approver_reason = validate_approver_title(row, cdd_df, question_col, answer_col, org_chart_text, session_logger)
        if not is_valid_approver:
            missing_titles.append({"user_id": user_id, "reason": approver_reason})

        # Determine if SOD check is needed for this user
        perform_sod_check = True
        if sod_by_management:
            try:
                sod_check_col = "SOD Check Performed by Mgmt"
                if sod_check_col in test_df.columns and pd.notna(row[sod_check_col]):
                    if normalize_text(row[sod_check_col]) == "yes":
                        perform_sod_check = False
                        session_logger.log(
                            "Audit Processing",
                            f"SOD Check Skipped: User ID={user_id}",
                            decision="Skipped",
                            reason="SOD check performed by management",
                            source=f"SOD Check Performed by Mgmt={row[sod_check_col]}"
                        )
                else:
                    session_logger.log(
                        "Audit Processing",
                        f"SOD Check Validation: User ID={user_id}",
                        decision="Error",
                        reason="Answer to SOD checked or not is missing",
                        source=f"Column={sod_check_col}, Value={row.get(sod_check_col, 'N/A')}"
                    )
            except Exception as e:
                session_logger.log(
                    "Audit Processing",
                    f"SOD Check Validation: User ID={user_id}",
                    decision="Error",
                    reason=f"Error checking SOD column: {str(e)}",
                    source=f"Column={sod_check_col}"
                )

        # Prepare data for SOD conflict check if needed
        sod_result = None
        if perform_sod_check:
            ticket_row = {
                "User ID": user_id,
                "Privileges": privileges
            }
            sod_results = check_all_user_conflicts_from_tickets([ticket_row], controller, privilege_aliases)
            sod_result = next((r for r in sod_results if r["user_token"] == user_id), None)

        # Apply relevant tests based on classification
        applicable_tests = [
            test for test in control_tests
            if (test["applies_to"] == "new" and classification["is_new_user"]) or
               (test["applies_to"] == "modified" and classification["is_modified_user"]) or
               (test["applies_to"] == "both")
        ]

        for test in applicable_tests:
            test_name = f"{test['name']}"
            test_type = test["type"]
            try:
                if test_type == "Access approval":
                    if not user_name or not approver_name:
                        result_status = "FAIL"
                        reason = f"Validation failed for User ID {user_id}: Missing User Name='{user_name}' or Approver Name='{approver_name}'"
                        session_logger.log(
                            "Audit Processing",
                            f"Self-Approval Check: User ID={user_id}",
                            decision="Rejected",
                            reason=reason,
                            source=f"User Name={user_name}, Approver Name={approver_name}"
                        )
                    elif normalize_text(user_name) == normalize_text(approver_name):
                        result_status = "FAIL"
                        reason = f"Validation failed for User ID {user_id}: Self Approval detected (User Name='{user_name}' is the same as Approver Name='{approver_name}')"
                        session_logger.log(
                            "Audit Processing",
                            f"Self-Approval Check: User ID={user_id}",
                            decision="Rejected",
                            reason=reason,
                            source=f"User Name={user_name}, Approver Name={approver_name}"
                        )
                    else:
                        result_status = "PASS" if is_valid_approver else "FAIL"
                        reason = approver_reason
                        session_logger.log(
                            "Audit Processing",
                            f"Self-Approval Check: User ID={user_id}",
                            decision="Accepted",
                            reason=f"No self-approval detected: User Name='{user_name}' is different from Approver Name='{approver_name}'",
                            source=f"User Name={user_name}, Approver Name={approver_name}"
                        )
                elif test_type == "Access requested = Access granted":
                    if normalize_text(access_requested) == normalize_text(access_granted):
                        result_status = "PASS"
                        reason = f"Validation passed for User ID {user_id}: Requested Access='{access_requested}' exactly matches Granted Access='{access_granted}'"
                    else:
                        query = f"Is the requested access '{access_requested}' consistent with the granted access '{access_granted}' for User ID {user_id}?"
                        result = cached_qa_invoke(query, cdd_text) if qa_chain is not None else "yes"
                        is_pass = "yes" in result.lower() or "consistent" in result.lower()
                        result_status = "PASS" if is_pass else "FAIL"
                        reason = f"Validation {'passed' if is_pass else 'failed'} for User ID {user_id}: Requested Access='{access_requested}', Granted Access='{access_granted}', {'LLM Response: ' + result if qa_chain is not None else 'No QA chain, assuming consistent'}"
                elif test_type == "Access appropriateness":
                    if not perform_sod_check:
                        result_status = "SKIPPED"
                        reason = f"Validation skipped for User ID {user_id}: SOD check performed by management"
                    elif not privileges:
                        result_status = "FAIL"
                        reason = f"Validation failed for User ID {user_id}: No privileges provided"
                    elif not sod_result:
                        result_status = "FAIL"
                        reason = f"Validation failed for User ID {user_id}: Error processing SOD conflicts"
                    else:
                        conflict_exists = sod_result["conflict_exists"]
                        unknown_privileges = sod_result["unknown_privileges"]
                        if conflict_exists:
                            result_status = "FAIL"
                            reason = f"Validation failed for User ID {user_id}: SoD Conflict detected"
                        elif unknown_privileges:
                            result_status = "FAIL"
                            reason = f"Validation failed for User ID {user_id}: Unknown privileges: {', '.join(unknown_privileges)}"
                        else:
                            result_status = "PASS"
                            reason = f"Validation passed for User ID {user_id}: No SOD conflicts detected in privileges: {privileges}"
                elif test_type == "Approver = Granter/Creator/Provisioner":
                    if not approver_name or not granter:
                        result_status = "FAIL"
                        reason = f"Validation failed for User ID {user_id}: Missing Approver Name='{approver_name}' or Granter='{granter}'"
                    else:
                        is_pass = normalize_text(approver_name) != normalize_text(granter)
                        result_status = "PASS" if is_pass else "FAIL"
                        reason = f"Validation {'passed' if is_pass else 'failed'} for User ID {user_id}: Approver Name='{approver_name}', Granter Name='{granter}', {'Approver and granter are different' if is_pass else 'Approver and granter are the same'}"
                elif test_type == "Date validation":
                    approval_date = row.get("Access Approval Date", None)
                    created_date = row.get("Date of Access Created", None)
                    modified_date = row.get("Date of Last Modified", None)
                    try:
                        approval_date = pd.to_datetime(approval_date, errors='coerce') if approval_date and pd.notna(approval_date) else None
                        created_date = pd.to_datetime(created_date, errors='coerce') if created_date and pd.notna(created_date) else None
                        modified_date = pd.to_datetime(modified_date, errors='coerce') if modified_date and pd.notna(modified_date) else None

                        if not approval_date or (not created_date and not modified_date):
                            result_status = "FAIL"
                            reason = f"Validation failed for User ID {user_id}: Missing Access Approval Date='{approval_date}' or both Date of Access Created='{created_date}' and Date of Last Modified='{modified_date}'"
                        else:
                            created_diff = abs((approval_date - created_date).days) if created_date else float('inf')
                            modified_diff = abs((approval_date - modified_date).days) if modified_date else float('inf')
                            
                            closer_date = created_date if created_diff <= modified_diff else modified_date
                            closer_date_name = "Date of Access Created" if created_diff <= modified_diff else "Date of Last Modified"
                            closer_date_str = closer_date.strftime('%d/%m/%Y') if closer_date else "N/A"

                            if closer_date and closer_date < approval_date:
                                result_status = "FAIL"
                                reason = f"Validation failed for User ID {user_id}: {closer_date_name}='{closer_date_str}' is before Access Approval Date='{approval_date.strftime('%d/%m/%Y')}'"
                            else:
                                result_status = "PASS"
                                reason = f"Validation passed for User ID {user_id}: {closer_date_name}='{closer_date_str}' is on or after Access Approval Date='{approval_date.strftime('%d/%m/%Y')}'"

                        session_logger.log(
                            "Audit Processing",
                            f"Date Validation: User ID={user_id}",
                            decision=result_status,
                            reason=reason,
                            source=f"Access Approval Date={approval_date}, Date of Access Created={created_date}, Date of Last Modified={modified_date}"
                        )
                    except Exception as e:
                        result_status = "FAIL"
                        reason = f"Validation failed for User ID {user_id}: Error processing dates: {str(e)}"
                        session_logger.log(
                            "Audit Processing",
                            f"Date Validation: User ID={user_id}",
                            decision="Error",
                            reason=reason,
                            source=f"Access Approval Date={approval_date}, Date of Access Created={created_date}, Date of Last Modified={modified_date}"
                        )

                results.append({
                    "Test Name": test_name,
                    "Result": result_status,
                    "Reason": reason,
                    "User ID": user_id
                })

                session_logger.log(
                    "Audit Processing",
                    f"Test: {test_name} (User ID: {user_id})",
                    decision=result_status,
                    reason=reason,
                    source=f"User ID={user_id}"
                )
            except Exception as e:
                results.append({
                    "Test Name": test_name,
                    "Result": "FAIL",
                    "Reason": f"Validation failed for User ID {user_id}: Error processing test: {str(e)}",
                    "User ID": user_id
                })
                session_logger.log(
                    "Audit Processing",
                    f"Test: {test_name} (User ID: {user_id})",
                    decision="Error",
                    reason=f"Error processing test: {str(e)}",
                    source=f"User ID={user_id}"
                )

    # Group results by User ID and display in expanders
    if results or user_classifications:
        results_df = pd.DataFrame(results)
        grouped_results = results_df.groupby("User ID")
        
        for user_id, group_df in grouped_results:
            with st.expander(f"User ID: {user_id}"):
                classification = next((c for c in user_classifications if c["User ID"] == user_id), None)
                if classification:
                    st.markdown(f"**Status**: {classification['Status']}")
                    st.markdown(f"**Reason**: {classification['Reason']}")
                else:
                    st.markdown("**Status**: Unknown")
                    st.markdown("**Reason**: Classification not available")
                
                display_df = group_df.drop(columns=["User ID"], errors="ignore")
                st.dataframe(display_df, use_container_width=True)
        
        session_logger.log(
            "Audit Processing",
            "Results Display",
            decision="Accepted",
            reason=f"Displayed {len(results)} audit results across {len(grouped_results)} users"
        )
    else:
        st.warning("⚠️ No audit results generated. Please ensure testing data is valid.")
        session_logger.log(
            "Audit Processing",
            "Results Display",
            decision="Rejected",
            reason="No audit results generated"
        )

if __name__ == "__main__":
    process_audit()