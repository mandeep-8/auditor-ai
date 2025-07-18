# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "inputs")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
COLLECTION_NAME = "Audit-Gov-Doc"

# Ensure directories exist
for dir_path in [INPUT_DIR, UPLOAD_DIR, OUTPUT_DIR, LOG_DIR, SESSIONS_DIR, INPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

