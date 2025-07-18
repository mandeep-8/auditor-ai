#app.py
import streamlit as st
from datetime import datetime
from components.pip_input import show_pip_input_form
from utils.session_logger import get_session_logger
from config import LOG_DIR, OUTPUT_DIR

# Set up the Streamlit app's basic configuration
st.set_page_config(page_title="AI Auditor", layout="wide")

# Display the app's title
st.title("🕵️‍♂️📊 AI Auditor - Preliminary Information Pack Input")

# Initialize session state (like a shared memory for the app)
if "session_id" not in st.session_state or st.session_state.session_id == "":
    # Create a unique ID for the session based on the current time
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Initialize variables to store validation results and other data
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

# Show the input form for preliminary information
form_result = show_pip_input_form()