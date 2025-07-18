# utils/session_logger.py
import os
import json
from datetime import datetime
import uuid
import hashlib
from langsmith import Client
from langsmith.run_helpers import traceable, get_current_run_tree
from dotenv import load_dotenv

load_dotenv()

class SessionLogger:
    """Handles logging to a single JSON file per session with LangSmith integration."""
    
    def __init__(self, log_dir, session_id):
        """Initialize logger for a session."""
        self.session_id = session_id
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.json_log_file = os.path.join(log_dir, f"session_{session_id}.json")
        self.json_logs = []
        self.logged_events = set()  # Track unique events to avoid duplicates
        
        # Initialize LangSmith
        self.langsmith = Client(
            api_key=os.getenv("LANGCHAIN_API_KEY"),
            api_url=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        )
        
        self.load_json_logs()
    
    def load_json_logs(self):
        """Load existing JSON logs from file, if it exists."""
        if os.path.exists(self.json_log_file):
            try:
                with open(self.json_log_file, "r", encoding="utf-8") as f:
                    self.json_logs = json.load(f)
                    # Populate logged_events with hashes of existing logs
                    for log in self.json_logs:
                        event_hash = self._hash_event(log)
                        self.logged_events.add(event_hash)
            except (json.JSONDecodeError, IOError):
                self.json_logs = []
    
    def _hash_event(self, log_entry):
        """Generate a unique hash for a log entry to prevent duplicates."""
        key_fields = (
            log_entry.get("component", ""),
            log_entry.get("message", ""),
            log_entry.get("decision", ""),
            json.dumps(log_entry.get("context", {}), sort_keys=True)
        )
        return hashlib.md5("".join(str(f) for f in key_fields).encode()).hexdigest()
    
    def log(self, component, message, decision=None, reason=None, source=None, level="INFO", context=None, langsmith_metadata=None):
        """Log an event to JSON file with LangSmith integration."""
        # Check if the log level is sufficient
        log_levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
        current_level = log_levels.get(level.upper(), 20)
        env_level = log_levels.get(os.getenv("LOG_LEVEL", "INFO").upper(), 20)
        if current_level < env_level:
            return  # Skip logging if level is below the threshold
        
        context = context or {}
        if langsmith_metadata:
            context["langsmith_details"] = langsmith_metadata
        
        log_entry = {
            "log_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "component": component,
            "message": message,
            "level": level.upper()
        }
        if decision:
            log_entry["decision"] = decision
            log_entry["context"] = context
            log_entry["context"]["decision_id"] = f"{component.lower().replace(' ', '_')}_{str(uuid.uuid4())[:8]}"
        if reason:
            log_entry["reason"] = reason
        if source:
            log_entry["source"] = source
        
        # Check for duplicate events
        event_hash = self._hash_event(log_entry)
        if event_hash in self.logged_events:
            return  # Skip duplicate log
        
        self.json_logs.append(log_entry)
        self.logged_events.add(event_hash)
        
        # Save JSON logs
        try:
            with open(self.json_log_file, "w", encoding="utf-8") as f:
                json.dump(self.json_logs, f, indent=2)
        except (IOError, PermissionError) as e:
            print(f"Failed to save JSON log: {str(e)}")
    
    def get_logs(self):
        """Return the current JSON logs."""
        return self.json_logs
    
    @traceable(run_type="llm")
    def log_llm_call(self, component, message, prompt, response, model, start_time, end_time, context=None):
        """Log LLM call with LangSmith tracking."""
        context = context or {}
        duration_ms = (end_time - start_time) * 1000
        
        # Get the current run tree to extract the trace ID
        run_tree = get_current_run_tree()
        trace_id = str(run_tree.id) if run_tree else str(uuid.uuid4())  # Fallback to UUID if run_tree is None
        
        # Create metadata for LangSmith
        langsmith_metadata = {
            "trace_id": trace_id,
            "prompt": prompt,
            "response": response,
            "model": model,
            "response_time_ms": int(duration_ms)
        }
        
        # Add LangSmith metadata to context
        context["langsmith_details"] = langsmith_metadata
        
        self.log(
            component=component,
            message=message,
            decision="Answered" if response else "Error",
            reason="LLM call completed",
            source="LLM",
            level="DEBUG",
            context=context,
            langsmith_metadata=langsmith_metadata
        )

def get_session_logger(log_dir, session_id):
    """Get or create a SessionLogger instance."""
    return SessionLogger(log_dir, session_id)