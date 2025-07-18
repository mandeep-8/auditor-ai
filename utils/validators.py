#utils/validators.py
def validate_pip_inputs(name: str, title: str, email: str, org_chart_file):
    missing_fields = []

    if not name.strip():
        missing_fields.append("Control owner name")
    if not title.strip():
        missing_fields.append("Job title")
    if not email.strip():
        missing_fields.append("Email address")

    if missing_fields:
        missing_str = ", ".join(missing_fields)
        return False, f"❌ The following required fields are missing: {missing_str}."

    return True, "Valid inputs."
