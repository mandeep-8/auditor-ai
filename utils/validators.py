# utils/validators.py
from datetime import datetime

def validate_pip_inputs(name: str, title: str, email: str, org_chart_file, audit_period: str = None):
    missing_fields = []

    if not name.strip():
        missing_fields.append("Control owner name")
    if not title.strip():
        missing_fields.append("Job title")
    if not email.strip():
        missing_fields.append("Email address")
    if not audit_period or not audit_period.strip():
        missing_fields.append("Audit period")
    elif audit_period:
        try:
            # Trim whitespace and split
            audit_period = audit_period.strip()
            start_date_str, end_date_str = audit_period.split("-")
            start_date_str, end_date_str = start_date_str.strip(), end_date_str.strip()
            start_date = datetime.strptime(start_date_str, "%d/%m/%Y")
            end_date = datetime.strptime(end_date_str, "%d/%m/%Y")
            if start_date > end_date:
                return False, "❌ Audit period start date must be before end date."
        except ValueError:
            return False, "❌ Invalid audit period format. Use DD/MM/YYYY - DD/MM/YYYY (e.g., 01/01/2025 - 31/12/2025)."

    if missing_fields:
        missing_str = ", ".join(missing_fields)
        return False, f"❌ The following required fields are missing: {missing_str}."

    return True, "Valid inputs."