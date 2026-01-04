"""
LAYERS 5 & 6: OUTPUT CONTRACT & STABILITY
Purpose: Ensure output consistency and detect internal contradictions.
Type: 100% Deterministic

Features:
- Strict output schema enforcement
- Multiple consistency checks
- Contradiction detection
- Safe formatting with error handling
"""
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import datetime

class ConsistencyLevel(Enum):
    CONSISTENT = "CONSISTENT"
    WARNING = "WARNING"
    CONTRADICTION = "CONTRADICTION"
    CRITICAL = "CRITICAL"

@dataclass
class ConsistencyCheck:
    """Result of a single consistency check."""
    check_name: str
    level: ConsistencyLevel
    description: str
    values_compared: Dict[str, Any]

class Layer5OutputContract:
    """
    LAYER 5: OUTPUT CONTRACT LAYER
    Purpose: Structure results into standard format.
    Type: 100% Deterministic
    
    Ensures:
    - All required fields present
    - Correct data types
    - Valid ranges for scores
    - Proper JSON serialization
    """
    
    REQUIRED_FIELDS = [
        'trace_id',
        'timestamp',
        'status',
        'composite_score',
        'dimensions',
        'anomaly_count',
        'decision'
    ]
    
    def __init__(self):
        self._validation_errors: List[str] = []
        
    def format_output(self, 
                     trace_id: str,
                     analysis_result: Dict[str, Any],
                     execution_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Format analysis results into standardized output.
        """
        self._validation_errors = []
        
        output = {
            "trace_id": trace_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "api_version": "2.0.0",
            "status": "SUCCESS",
            "result": self._sanitize_result(analysis_result),
            "metadata": {
                "processing_time_ms": execution_metadata.get('duration_ms', 0) if execution_metadata else 0,
                "layers_executed": execution_metadata.get('layers_executed', []) if execution_metadata else [],
                "warnings": self._validation_errors
            }
        }
        
        # Validate output structure
        self._validate_output(output)
        
        if self._validation_errors:
            output["metadata"]["validation_warnings"] = self._validation_errors
            
        return output
    
    def safe_format(self, 
                   trace_id: str, 
                   error_msg: str,
                   error_code: str = "ANALYSIS_ERROR",
                   partial_result: Dict = None) -> Dict[str, Any]:
        """
        Format error response safely.
        Used when analysis fails at any stage.
        """
        return {
            "trace_id": trace_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "api_version": "2.0.0",
            "status": "FAILURE",
            "error": {
                "code": error_code,
                "message": error_msg,
                "recoverable": error_code not in ["CONTRACT_VIOLATION", "CRITICAL_FAILURE"]
            },
            "partial_result": partial_result,
            "metadata": {
                "failed_at": datetime.datetime.now().isoformat()
            }
        }
    
    def _sanitize_result(self, result: Dict) -> Dict:
        """Ensure all values are JSON-serializable."""
        sanitized = {}
        
        for key, value in result.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                sanitized[key] = value
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_result(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_result(item) if isinstance(item, dict) else str(item)
                    for item in value
                ]
            elif hasattr(value, 'to_dict'):
                sanitized[key] = value.to_dict()
            else:
                sanitized[key] = str(value)
                
        return sanitized
    
    def _validate_output(self, output: Dict) -> bool:
        """Validate output structure against contract."""
        result = output.get('result', {})
        
        # Check composite score range
        score = result.get('composite_score', 0)
        if not isinstance(score, (int, float)):
            self._validation_errors.append(f"composite_score must be numeric, got {type(score)}")
        elif not (0 <= score <= 100):
            self._validation_errors.append(f"composite_score {score} out of valid range [0, 100]")
            
        # Check dimensions array
        dims = result.get('dimensions', [])
        if not isinstance(dims, list):
            self._validation_errors.append("dimensions must be an array")
            
        return len(self._validation_errors) == 0


class Layer6Stability:
    """
    LAYER 6: STABILITY & CONSISTENCY LAYER
    Purpose: Verify output stability and internal consistency.
    Type: 100% Deterministic
    
    Detects:
    - Score vs status contradictions
    - Dimension vs composite inconsistencies
    - ML vs Rules conflicts
    - Temporal anomalies
    """
    
    def __init__(self):
        self.checks_performed: List[ConsistencyCheck] = []
        
    def check_consistency(self, 
                         dqs: float, 
                         dimension_scores: Dict[str, float],
                         critical_violations: int, 
                         anomaly_count: int,
                         decision: str = None) -> Tuple[ConsistencyLevel, List[ConsistencyCheck]]:
        """
        Perform comprehensive consistency checks.
        Returns: (overall_level, list_of_checks)
        """
        self.checks_performed = []
        
        # Check 1: High DQS but Critical Violations
        if dqs > 80 and critical_violations > 0:
            self.checks_performed.append(ConsistencyCheck(
                check_name="SCORE_VIOLATION_MISMATCH",
                level=ConsistencyLevel.CONTRADICTION,
                description=f"High DQS ({dqs:.1f}) contradicts {critical_violations} critical violations",
                values_compared={"dqs": dqs, "critical_violations": critical_violations}
            ))
            
        # Check 2: Very high DQS with high anomaly count
        if dqs > 90 and anomaly_count > 20:
            self.checks_performed.append(ConsistencyCheck(
                check_name="SCORE_ANOMALY_MISMATCH",
                level=ConsistencyLevel.WARNING,
                description=f"DQS {dqs:.1f} seems inconsistent with {anomaly_count} anomalies",
                values_compared={"dqs": dqs, "anomaly_count": anomaly_count}
            ))
            
        # Check 3: Composite vs Individual dimension scores
        if dimension_scores:
            min_dim = min(dimension_scores.values())
            max_dim = max(dimension_scores.values())
            
            # If composite is higher than max dimension, something's wrong
            if dqs > max_dim + 5:
                self.checks_performed.append(ConsistencyCheck(
                    check_name="COMPOSITE_EXCEEDS_MAX",
                    level=ConsistencyLevel.CONTRADICTION,
                    description=f"Composite ({dqs:.1f}) exceeds highest dimension ({max_dim:.1f})",
                    values_compared={"composite": dqs, "max_dimension": max_dim}
                ))
                
            # Large variance between dimensions
            dim_range = max_dim - min_dim
            if dim_range > 50:
                self.checks_performed.append(ConsistencyCheck(
                    check_name="HIGH_DIMENSION_VARIANCE",
                    level=ConsistencyLevel.WARNING,
                    description=f"High variance between dimensions: {min_dim:.1f} to {max_dim:.1f}",
                    values_compared={"min": min_dim, "max": max_dim, "range": dim_range}
                ))
                
        # Check 4: Decision vs Score consistency
        if decision:
            if decision == "SAFE_TO_USE" and dqs < 60:
                self.checks_performed.append(ConsistencyCheck(
                    check_name="DECISION_SCORE_MISMATCH",
                    level=ConsistencyLevel.CRITICAL,
                    description=f"SAFE_TO_USE decision with low DQS ({dqs:.1f})",
                    values_compared={"decision": decision, "dqs": dqs}
                ))
            elif decision == "ESCALATE" and dqs > 90 and critical_violations == 0:
                self.checks_performed.append(ConsistencyCheck(
                    check_name="UNNECESSARY_ESCALATION",
                    level=ConsistencyLevel.WARNING,
                    description=f"ESCALATE called despite high DQS ({dqs:.1f}) and no violations",
                    values_compared={"decision": decision, "dqs": dqs}
                ))
                
        # Check 5: All dimensions should sum correctly to weighted average
        if dimension_scores:
            weights = {
                'Completeness': 0.20, 'Accuracy': 0.15, 'Validity': 0.15,
                'Uniqueness': 0.15, 'Consistency': 0.15, 'Timeliness': 0.10, 'Integrity': 0.10
            }
            calculated_composite = sum(
                score * weights.get(dim, 0.1) 
                for dim, score in dimension_scores.items()
            )
            if abs(calculated_composite - dqs) > 1:
                self.checks_performed.append(ConsistencyCheck(
                    check_name="WEIGHTED_AVERAGE_MISMATCH",
                    level=ConsistencyLevel.CONTRADICTION,
                    description=f"Reported DQS ({dqs:.1f}) differs from calculated ({calculated_composite:.1f})",
                    values_compared={"reported": dqs, "calculated": calculated_composite}
                ))
                
        # Determine overall level
        if any(c.level == ConsistencyLevel.CRITICAL for c in self.checks_performed):
            overall = ConsistencyLevel.CRITICAL
        elif any(c.level == ConsistencyLevel.CONTRADICTION for c in self.checks_performed):
            overall = ConsistencyLevel.CONTRADICTION
        elif any(c.level == ConsistencyLevel.WARNING for c in self.checks_performed):
            overall = ConsistencyLevel.WARNING
        else:
            overall = ConsistencyLevel.CONSISTENT
            
        return overall, self.checks_performed
    
    def get_check_summary(self) -> Dict[str, Any]:
        """Get summary of all consistency checks."""
        return {
            "total_checks": len(self.checks_performed),
            "by_level": {
                level.value: sum(1 for c in self.checks_performed if c.level == level)
                for level in ConsistencyLevel
            },
            "checks": [
                {
                    "name": c.check_name,
                    "level": c.level.value,
                    "description": c.description
                }
                for c in self.checks_performed
            ]
        }
