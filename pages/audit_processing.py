import streamlit as st
import pandas as pd
import os
import json
from utils.question_answering import get_qa_chain
from utils.session_logger import get_session_logger
from utils.content_validator import extract_text_from_file
from config import INPUT_DIR, LOG_DIR, SESSIONS_DIR, OUTPUT_DIR
from functools import lru_cache

# Cache QA chain results to avoid redundant calls
@lru_cache(maxsize=1000)
def cached_qa_invoke(query, context):
    """Cache QA chain results to improve performance."""
    qa_chain = st.session_state.qa_chain
    return qa_chain.invoke({"query": query, "context": context})["result"].strip()

# SOD conflict detection data
privilege_aliases = {
    "Granting/Modifying  User Access": 1,
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
        privilege_texts = [p.strip() for p in privileges_raw.split(',')] if privileges_raw else []

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

    # Step 1: Check if approver title matches CDD answer (only if QA chain is available)
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
            query = f"Is the job title {approver_title} considered a managerial level position in the organizational chart?"
            result = cached_qa_invoke(query, org_chart_text) if st.session_state.get("qa_chain") is not None else "yes"
            is_managerial = "yes" in result.lower() or "manager" in result.lower() or "director" in result.lower() or "executive" in result.lower()
            if is_managerial:
                session_logger.log(
                    "Audit Processing",
                    f"Approver Title Validation: User ID={user_id}",
                    decision="Accepted",
                    reason=f"Approver title {approver_title} is managerial in organizational chart",
                    source=f"Org Chart Response={result}"
                )
                return True, f"Validation passed for User ID {user_id}: Approver Title='{approver_title}' is managerial per organizational chart: {result}"
            else:
                session_logger.log(
                    "Audit Processing",
                    f"Approver Title Validation: User ID={user_id}",
                    decision="Rejected",
                    reason=f"Approver title {approver_title} not managerial in organizational chart",
                    source=f"Org Chart Response={result}"
                )
                return False, f"Validation failed for User ID {user_id}: Approver Title='{approver_title}' not managerial per organizational chart: {result}"
        except Exception as e:
            session_logger.log(
                "Audit Processing",
                f"Approver Title Validation: User ID={user_id}",
                decision="Error",
                reason=f"Org chart title check error: {str(e)}",
                source=f"Approver={approver_name}"
            )
            return False, f"Validation failed for User ID {user_id}: Error checking Approver Title='{approver_title}' in organizational chart: {str(e)}"
    else:
        # If no QA chain and no org chart, assume valid if line manager check passes or no other checks fail
        session_logger.log(
            "Audit Processing",
            f"Approver Title Validation: User ID={user_id}",
            decision="Accepted",
            reason="No QA chain or org chart available, assuming valid approver title",
            source=f"Approver={approver_name}"
        )
        return True, f"Validation passed for User ID {user_id}: No QA chain or org chart available, assuming Approver Name='{approver_name}', Approver Title='{approver_title}' is valid"

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
        "user name": "Role",
        "provisioner name": "Granter",
        "privileges": "Privileges",
        "privileges granted": "Privileges",
        "sod check performed by mgmt": "SOD Check Performed by Mgmt"
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

def process_audit():
    """Perform control tests and display results with approver title validation."""
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
    cdd_df = st.session_state.excel_df

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
        # Assume question 26 is the last row in the CDD Excel
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

    # Try loading each org chart file from uploads directory first, then INPUT_DIR
    for org_chart_filename in org_chart_filenames:
        # First try uploads directory
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
        else:
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
    required_columns = ["User ID", "Approver Name", "Approver Title", "Line Manager", "Access Requested", "Access Granted", "Role", "Granter", "Privileges"]
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
        {"name": "New user access approved", "type": "Access approval"},
        {"name": "Modified user access approved", "type": "Access approval"},
        {"name": "New user access consistent", "type": "Access requested = Access granted"},
        {"name": "Modified user access consistent", "type": "Access requested = Access granted"},
        {"name": "New user access appropriate", "type": "Access is appropriate"},
        {"name": "Modified user access appropriate", "type": "Access is appropriate"},
        {"name": "SoD: New user access", "type": "Approver = Granter/Creater"},
        {"name": "SoD: Modified user access", "type": "Approver = Granter/Creater"},
        {"name": "SoD: New user privilege conflicts", "type": "Privilege conflicts"},
        {"name": "SoD: Modified user privilege conflicts", "type": "Privilege conflicts"}
    ]

    # Process rows and control tests in a single loop
    st.write("### Audit Results")
    results = []
    missing_titles = []
    for idx, row in test_df.iterrows():
        user_id = row["User ID"]
        access_requested = row.get("Access Requested", "")
        access_granted = row.get("Access Granted", "")
        role = row.get("Role", "")
        approver_name = row.get("Approver Name", "")
        granter = row.get("Granter", "")
        privileges = row.get("Privileges", "")

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

        for test in control_tests:
            test_name = f"{test['name']}"
            test_type = test["type"]
            try:
                if test_type == "Access approval":
                    result_status = "PASS" if is_valid_approver else "FAIL"
                    reason = approver_reason
                elif test_type == "Access requested = Access granted":
                    # Pre-check exact match
                    if normalize_text(access_requested) == normalize_text(access_granted):
                        result_status = "PASS"
                        reason = f"Validation passed for User ID {user_id}: Requested Access='{access_requested}' exactly matches Granted Access='{access_granted}'"
                    else:
                        query = f"Is the requested access '{access_requested}' consistent with the granted access '{access_granted}' for User ID {user_id}?"
                        result = cached_qa_invoke(query, cdd_text) if qa_chain is not None else "yes"
                        is_pass = "yes" in result.lower() or "consistent" in result.lower()
                        result_status = "PASS" if is_pass else "FAIL"
                        reason = f"Validation {'passed' if is_pass else 'failed'} for User ID {user_id}: Requested Access='{access_requested}', Granted Access='{access_granted}', {'LLM Response: ' + result if qa_chain is not None else 'No QA chain, assuming consistent'}"
                elif test_type == "Access is appropriate":
                    query = f"Is the access '{access_granted}' for User ID {user_id} commensurate with the role '{role}' and free of segregation of duties conflicts?"
                    result = cached_qa_invoke(query, cdd_text) if qa_chain is not None else "yes"
                    is_pass = "yes" in result.lower() or "appropriate" in result.lower() or "no conflict" in result.lower()
                    result_status = "PASS" if is_pass else "FAIL"
                    reason = f"Validation {'passed' if is_pass else 'failed'} for User ID {user_id}: Granted Access='{access_granted}', Role='{role}', {'LLM Response: ' + result if qa_chain is not None else 'No QA chain, assuming appropriate'}"
                elif test_type == "Approver = Granter/Creater":
                    # Pre-check presence of approver_name and granter
                    if not approver_name or not granter:
                        result_status = "FAIL"
                        reason = f"Validation failed for User ID {user_id}: Missing Approver Name='{approver_name}' or Granter='{granter}'"
                    else:
                        is_pass = normalize_text(approver_name) != normalize_text(granter)
                        result_status = "PASS" if is_pass else "FAIL"
                        reason = f"Validation {'passed' if is_pass else 'failed'} for User ID {user_id}: Approver Name='{approver_name}', Granter Name='{granter}', {'Approver and granter are different' if is_pass else 'Approver and granter are the same'}"
                elif test_type == "Privilege conflicts":
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
                        conflicts = sod_result["conflicting_privileges"]
                        unknown_privileges = sod_result["unknown_privileges"]
                        if conflict_exists:
                            conflict_details = ", ".join(
                                [f"{c['privilege_1_name']} vs {c['privilege_2_name']}" for c in conflicts]
                            )
                            result_status = "FAIL"
                            reason = f"Validation failed for User ID {user_id}: SOD conflicts detected: {conflict_details}"
                        elif unknown_privileges:
                            result_status = "FAIL"
                            reason = f"Validation failed for User ID {user_id}: Unknown privileges: {', '.join(unknown_privileges)}"
                        else:
                            result_status = "PASS"
                            reason = f"Validation passed for User ID {user_id}: No SOD conflicts detected in privileges: {privileges}"

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
    if results:
        results_df = pd.DataFrame(results)
        grouped_results = results_df.groupby("User ID")
        
        for user_id, group_df in grouped_results:
            with st.expander(f"User ID: {user_id}"):
                display_df = group_df.drop(columns=["User ID", "Approver Status"], errors="ignore")
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