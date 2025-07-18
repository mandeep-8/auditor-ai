#pages/pip_report.py
import streamlit as st
import os
from fpdf import FPDF
from config import OUTPUT_DIR
from utils.session_logger import get_session_logger

st.set_page_config(page_title="PIP Report", layout="centered", initial_sidebar_state="collapsed")
st.title("📄 PIP Report")

def generate_pdf_report(content_lines, pip_info_lines, pdf_path, session_logger):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "PIP Report Summary", ln=True)

    # Uploaded Files
    pdf.set_font("Arial", "B", 12)
    pdf.ln(5)
    pdf.cell(0, 10, "Uploaded Files:", ln=True)
    pdf.set_font("Arial", "", 11)
    for line in pip_info_lines:
        clean_line = line.strip()
        if clean_line:
            pdf.multi_cell(0, 10, f"- {clean_line}")

    # Remarks
    pdf.set_font("Arial", "B", 12)
    pdf.ln(5)
    pdf.cell(0, 10, "Remarks:", ln=True)
    pdf.set_font("Arial", "", 11)
    for line in content_lines:
        clean_line = line.strip()
        if clean_line:
            pdf.multi_cell(0, 10, f"- {clean_line}")

    try:
        pdf.output(pdf_path)
        session_logger.log(
            "PIP Report",
            f"Generated PDF: {os.path.basename(pdf_path)}",
            decision="Accepted",
            reason="PDF report generated successfully",
            source=f"Files={pip_info_lines}, Remarks={content_lines}"
        )
    except (IOError, PermissionError) as e:
        session_logger.log(
            "PIP Report",
            f"Generating PDF: {os.path.basename(pdf_path)}",
            decision="Rejected",
            reason=f"Failed to generate PDF: {str(e)}"
        )
        raise

# Main logic
session_logger = get_session_logger(OUTPUT_DIR, st.session_state.session_id)
conclusion_path = os.path.join(OUTPUT_DIR, "pip_conclusion.txt")
pip_info_path = os.path.join(OUTPUT_DIR, "pip_info.txt")
pdf_output_path = os.path.join(OUTPUT_DIR, "pip_report.pdf")

if os.path.exists(conclusion_path) and os.path.exists(pip_info_path):
    with open(conclusion_path, "r", encoding="utf-8") as file:
        conclusion_lines = file.readlines()
    with open(pip_info_path, "r", encoding="utf-8") as file:
        pip_info_lines = file.readlines()

    session_logger.log(
        "PIP Report",
        "Loading Files",
        decision="Accepted",
        reason="pip_conclusion.txt and pip_info.txt loaded successfully",
        source=f"Files={['pip_conclusion.txt', 'pip_info.txt']}"
    )

    st.markdown("### 📌 Uploaded Files:")
    for idx, line in enumerate(pip_info_lines, start=1):
        st.markdown(f"{idx}. {line.strip()}")

    st.markdown("### 📌 Remarks:")
    for idx, line in enumerate(conclusion_lines, start=1):
        st.markdown(f"{idx}. {line.strip()}")

    # Generate PDF and display download button
    generate_pdf_report(conclusion_lines, pip_info_lines, pdf_output_path, session_logger)
    with open(pdf_output_path, "rb") as f:
        st.download_button(
            label="⬇️ Download PIP Report as PDF",
            data=f,
            file_name="pip_report.pdf",
            mime="application/pdf"
        )
else:
    st.warning("⚠️ Submit the PIP form to generate the report.")
    session_logger.log(
        "PIP Report",
        "Loading Files",
        decision="Rejected",
        reason="pip_conclusion.txt or pip_info.txt not found"
    )