"""
LAYER 11: LOGGING & TRACE LAYER
Purpose: Immutable audit trail with structured logging for ALL events.
Type: Infrastructure
Features:
- Unique trace_id per analysis run
- Severity levels (DEBUG, INFO, WARN, ERROR, CRITICAL)
- Performance timing
- Structured JSON output
"""
import json
import datetime
import uuid
from typing import Dict, Any, List, Optional
from enum import Enum

class LogSeverity(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class Layer11Logging:
    """
    Production-grade logging layer with immutable audit trail.
    """
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
        self.trace_id = f"TR_{uuid.uuid4().hex[:12].upper()}"
        self.start_time = datetime.datetime.now()
        self.layer_timings: Dict[str, float] = {}
        self._current_layer_start: Optional[datetime.datetime] = None
        
    def log_event(self, layer: str, status: str, details: Any, 
                  severity: LogSeverity = LogSeverity.INFO) -> None:
        """Log a single event with full context."""
        now = datetime.datetime.now()
        elapsed_ms = (now - self.start_time).total_seconds() * 1000
        
        entry = {
            "trace_id": self.trace_id,
            "timestamp": now.isoformat(),
            "elapsed_ms": round(elapsed_ms, 2),
            "layer": layer,
            "status": status,
            "severity": severity.value,
            "details": self._serialize_details(details)
        }
        self.logs.append(entry)
        
    def start_layer(self, layer_name: str) -> None:
        """Mark the start of a layer for timing purposes."""
        self._current_layer_start = datetime.datetime.now()
        self.log_event(layer_name, "STARTED", {}, LogSeverity.DEBUG)
        
    def end_layer(self, layer_name: str, status: str, details: Any) -> None:
        """End a layer and record timing."""
        if self._current_layer_start:
            duration = (datetime.datetime.now() - self._current_layer_start).total_seconds() * 1000
            self.layer_timings[layer_name] = round(duration, 2)
            details_with_timing = {
                "result": details,
                "duration_ms": round(duration, 2)
            }
            self.log_event(layer_name, status, details_with_timing)
        else:
            self.log_event(layer_name, status, details)
        self._current_layer_start = None
        
    def log_error(self, layer: str, error: Exception) -> None:
        """Log an error with full traceback context."""
        self.log_event(layer, "ERROR", {
            "error_type": type(error).__name__,
            "error_message": str(error)
        }, LogSeverity.ERROR)
        
    def log_critical(self, layer: str, message: str, context: Dict = None) -> None:
        """Log a critical failure that should trigger immediate attention."""
        self.log_event(layer, "CRITICAL", {
            "message": message,
            "context": context or {}
        }, LogSeverity.CRITICAL)
        
    def get_logs(self) -> List[Dict]:
        """Return all logs for this trace."""
        return self.logs
    
    def get_summary(self) -> Dict:
        """Return a summary of the entire analysis run."""
        total_time = (datetime.datetime.now() - self.start_time).total_seconds() * 1000
        error_count = sum(1 for log in self.logs if log['severity'] in ['ERROR', 'CRITICAL'])
        warn_count = sum(1 for log in self.logs if log['severity'] == 'WARN')
        
        return {
            "trace_id": self.trace_id,
            "total_duration_ms": round(total_time, 2),
            "layer_timings": self.layer_timings,
            "total_events": len(self.logs),
            "error_count": error_count,
            "warning_count": warn_count,
            "status": "HEALTHY" if error_count == 0 else "DEGRADED" if error_count < 3 else "CRITICAL"
        }
    
    def _serialize_details(self, details: Any) -> Any:
        """Safely serialize details for JSON output."""
        if isinstance(details, (str, int, float, bool, type(None))):
            return details
        if isinstance(details, dict):
            return {k: self._serialize_details(v) for k, v in details.items()}
        if isinstance(details, list):
            return [self._serialize_details(i) for i in details]
        return str(details)
    
    def export_json(self) -> str:
        """Export full trace as JSON string."""
        return json.dumps({
            "summary": self.get_summary(),
            "events": self.logs
        }, indent=2)
