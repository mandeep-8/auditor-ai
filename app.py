import streamlit as st
from datetime import datetime
from components.pip_input import show_pip_input_form
from utils.session_logger import get_session_logger
from config import LOG_DIR, OUTPUT_DIR

# Set up the Streamlit app's basic configuration
st.set_page_config(page_title="AI Auditor", layout="wide")

# Initialize session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "session_id" not in st.session_state or st.session_state.session_id == "":
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.org_chart_validation = {"results": [], "validated": False}
    st.session_state.governance_doc_validation = {"results": [], "validated": False}
    st.session_state.qa_chain = None
    st.session_state.questions_processed = False
    st.session_state.cdd_saved = False
    st.session_state.pip_control_owner_name = ""
    st.session_state.pip_control_owner_title = ""
    st.session_state.pip_submitted = False

    # Set up logging for the session
    session_logger = get_session_logger(LOG_DIR, st.session_state.session_id)
    session_logger.log(
        component="Application",
        message="Application session initialized",
        decision="Accepted",
        reason=f"New session ID: {st.session_state.session_id}",
        level="INFO",
        context={"session_id": st.session_state.session_id}
    )

# Display login form if not logged in
if not st.session_state.logged_in:
    st.title("🔐 AI Auditor Login")
    with st.form("login_form"):
        st.subheader("Please enter your credentials")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            # Hardcoded credentials for demonstration (replace with secure auth in production)
            if username == "admin" and password == "securepassword123":
                st.session_state.logged_in = True
                session_logger = get_session_logger(LOG_DIR, st.session_state.session_id)
                session_logger.log(
                    component="Authentication",
                    message="Login attempt",
                    decision="Accepted",
                    reason="Valid credentials provided",
                    level="INFO",
                    context={"username": username}
                )
                st.rerun()  # Refresh to show main app
            else:
                st.error("❌ Invalid username or password")
                session_logger = get_session_logger(LOG_DIR, st.session_state.session_id)
                session_logger.log(
                    component="Authentication",
                    message="Login attempt",
                    decision="Rejected",
                    reason="Invalid credentials",
                    level="ERROR",
                    context={"username": username}
                )
else:
    # Main app content
    st.title("🕵️‍♂️📊 AI Auditor - Preliminary Information Pack Input")
    form_result = show_pip_input_form()