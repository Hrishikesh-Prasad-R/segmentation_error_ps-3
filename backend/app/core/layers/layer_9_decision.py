"""
LAYER 9: DECISION GATE LAYER
Purpose: Final decision routing through Finite State Machine.
Type: 100% Deterministic

Features:
- Clear decision states (SAFE_TO_USE, REVIEW_REQUIRED, ESCALATE, NO_ACTION)
- Multi-input decision matrix
- Detailed rationale for each decision
- Audit-ready decision logs
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

class DecisionAction(Enum):
    SAFE_TO_USE = "SAFE_TO_USE"          # Automated processing OK
    REVIEW_REQUIRED = "REVIEW_REQUIRED"   # Human review needed
    ESCALATE = "ESCALATE"                 # Senior/specialist attention
    NO_ACTION = "NO_ACTION"               # Cannot proceed, data issues

class DecisionPriority(Enum):
    P1_CRITICAL = 1    # Immediate attention
    P2_HIGH = 2        # Same-day review
    P3_MEDIUM = 3      # Next business day
    P4_LOW = 4         # Within week
    P5_INFO = 5        # For information only

@dataclass
class DecisionFactor:
    """A factor that influenced the decision."""
    factor_name: str
    factor_value: Any
    weight: float
    contribution: str  # How this affected the decision

@dataclass
class DecisionResult:
    """Complete decision output."""
    action: DecisionAction
    priority: DecisionPriority
    rationale: str
    factors: List[DecisionFactor]
    confidence_band: str
    responsible_party: str
    next_steps: List[str]
    estimated_resolution_time: str
    
    def to_dict(self) -> Dict:
        return {
            "action": self.action.value,
            "priority": self.priority.name,
            "rationale": self.rationale,
            "factors": [
                {
                    "name": f.factor_name,
                    "value": str(f.factor_value),
                    "weight": f.weight,
                    "contribution": f.contribution
                }
                for f in self.factors
            ],
            "confidence_band": self.confidence_band,
            "responsible_party": self.responsible_party,
            "next_steps": self.next_steps,
            "estimated_resolution_time": self.estimated_resolution_time
        }

class Layer9DecisionGate:
    """
    LAYER 9: DECISION GATE LAYER
    Purpose: Map all signals to a final decision.
    Type: 100% Deterministic (Finite State Machine)
    
    Decision Matrix:
    - DQS Score thresholds
    - Anomaly counts
    - Conflict states
    - Confidence bands
    - Override conditions
    """
    
    # Thresholds (configurable)
    DQS_SAFE_THRESHOLD = 75.0
    DQS_REVIEW_THRESHOLD = 50.0
    DQS_ESCALATE_THRESHOLD = 30.0
    
    ANOMALY_SAFE_THRESHOLD = 5
    ANOMALY_REVIEW_THRESHOLD = 15
    
    def __init__(self):
        self.decision_log: List[str] = []
        
    def decide(self,
              dqs_composite: float,
              dimension_scores: Dict[str, float],
              anomaly_flags: List[Dict],
              confidence_band: str,
              conflict_result: Dict[str, Any],
              semantic_critical_violations: int = 0,
              ml_degraded: bool = False) -> DecisionResult:
        """
        Execute the decision state machine.
        Returns: Complete decision result.
        """
        self.decision_log = []
        factors: List[DecisionFactor] = []
        
        # =====================================================
        # PHASE 1: CRITICAL OVERRIDE CHECKS
        # These can force a decision regardless of other factors
        # =====================================================
        
        # Override 1: Critical semantic violations
        if semantic_critical_violations > 0:
            self.decision_log.append(f"OVERRIDE: {semantic_critical_violations} critical semantic violations")
            
            factors.append(DecisionFactor(
                factor_name="Critical Semantic Violations",
                factor_value=semantic_critical_violations,
                weight=1.0,
                contribution="FORCED ESCALATION"
            ))
            
            return DecisionResult(
                action=DecisionAction.ESCALATE,
                priority=DecisionPriority.P1_CRITICAL,
                rationale=f"Critical semantic violations ({semantic_critical_violations}) require immediate investigation. Business rules violated.",
                factors=factors,
                confidence_band=confidence_band,
                responsible_party="Data Governance Lead + Business Stakeholder",
                next_steps=[
                    "Identify specific violation records",
                    "Determine root cause (data entry, system, or fraud)",
                    "Remediate or reject entire batch",
                    "Document incident for audit"
                ],
                estimated_resolution_time="2-4 hours"
            )
            
        # Override 2: Catastrophically low DQS
        if dqs_composite < self.DQS_ESCALATE_THRESHOLD:
            self.decision_log.append(f"OVERRIDE: DQS ({dqs_composite:.1f}) below critical threshold")
            
            factors.append(DecisionFactor(
                factor_name="Composite DQS",
                factor_value=dqs_composite,
                weight=1.0,
                contribution="FORCED ESCALATION due to quality collapse"
            ))
            
            return DecisionResult(
                action=DecisionAction.ESCALATE,
                priority=DecisionPriority.P1_CRITICAL,
                rationale=f"DQS score ({dqs_composite:.1f}%) critically low. Data quality catastrophe.",
                factors=factors,
                confidence_band=confidence_band,
                responsible_party="Data Engineering Team + Source System Owner",
                next_steps=[
                    "Halt downstream processing",
                    "Investigate data source",
                    "Check for system failures or corruption",
                    "Full data refresh required"
                ],
                estimated_resolution_time="4-8 hours"
            )
            
        # Override 3: Conflict escalation
        if conflict_result.get('escalation_required'):
            self.decision_log.append("OVERRIDE: Conflict resolution triggered escalation")
            
            factors.append(DecisionFactor(
                factor_name="Conflict Escalation",
                factor_value=True,
                weight=0.9,
                contribution="Unresolvable conflict detected"
            ))
            
            return DecisionResult(
                action=DecisionAction.ESCALATE,
                priority=DecisionPriority.P2_HIGH,
                rationale="Conflicting signals could not be automatically resolved. Human judgment required.",
                factors=factors,
                confidence_band=confidence_band,
                responsible_party="Data Quality Analyst",
                next_steps=[
                    "Review conflict details",
                    "Determine priority signal",
                    "Make manual determination",
                    "Document rationale"
                ],
                estimated_resolution_time="1-2 hours"
            )
            
        # =====================================================
        # PHASE 2: STANDARD DECISION MATRIX
        # =====================================================
        
        anomaly_count = len(anomaly_flags)
        high_severity_anomalies = sum(1 for f in anomaly_flags if f.get('severity') in ['HIGH', 'CRITICAL'])
        
        # Add standard factors
        factors.append(DecisionFactor(
            factor_name="Composite DQS",
            factor_value=dqs_composite,
            weight=0.4,
            contribution="Primary quality indicator"
        ))
        
        factors.append(DecisionFactor(
            factor_name="Anomaly Count",
            factor_value=anomaly_count,
            weight=0.2,
            contribution="ML-detected anomalies"
        ))
        
        factors.append(DecisionFactor(
            factor_name="Confidence Band",
            factor_value=confidence_band,
            weight=0.2,
            contribution="Decision confidence level"
        ))
        
        factors.append(DecisionFactor(
            factor_name="Conflict Count",
            factor_value=conflict_result.get('total_conflicts', 0),
            weight=0.1,
            contribution="Signal alignment measure"
        ))
        
        factors.append(DecisionFactor(
            factor_name="ML Degraded",
            factor_value=ml_degraded,
            weight=0.1,
            contribution="System health indicator"
        ))
        
        # =====================================================
        # DECISION LOGIC: State Machine
        # =====================================================
        
        # State 1: High Quality + Low Anomalies + High Confidence = SAFE
        if (dqs_composite >= self.DQS_SAFE_THRESHOLD and 
            anomaly_count <= self.ANOMALY_SAFE_THRESHOLD and
            confidence_band == "HIGH"):
            
            self.decision_log.append("SAFE: All quality indicators positive")
            
            return DecisionResult(
                action=DecisionAction.SAFE_TO_USE,
                priority=DecisionPriority.P5_INFO,
                rationale=f"Data quality excellent ({dqs_composite:.1f}%). {anomaly_count} anomalies within tolerance. High confidence.",
                factors=factors,
                confidence_band=confidence_band,
                responsible_party="System (Automated)",
                next_steps=[
                    "Proceed with automated processing",
                    "No human intervention required",
                    "Results logged for audit"
                ],
                estimated_resolution_time="Immediate"
            )
            
        # State 2: Good Quality but some anomalies or medium confidence = REVIEW
        if (dqs_composite >= self.DQS_REVIEW_THRESHOLD and
            (anomaly_count > self.ANOMALY_SAFE_THRESHOLD or confidence_band != "HIGH")):
            
            self.decision_log.append("REVIEW: Quality acceptable but flags present")
            
            # Determine priority based on severity
            if high_severity_anomalies > 5:
                priority = DecisionPriority.P2_HIGH
            elif anomaly_count > self.ANOMALY_REVIEW_THRESHOLD:
                priority = DecisionPriority.P3_MEDIUM
            else:
                priority = DecisionPriority.P4_LOW
                
            return DecisionResult(
                action=DecisionAction.REVIEW_REQUIRED,
                priority=priority,
                rationale=f"DQS acceptable ({dqs_composite:.1f}%) but {anomaly_count} anomalies flagged. Confidence: {confidence_band}.",
                factors=factors,
                confidence_band=confidence_band,
                responsible_party="Data Quality Reviewer",
                next_steps=[
                    "Review flagged anomalies",
                    "Verify critical records",
                    "Approve or reject after review",
                    "Document findings"
                ],
                estimated_resolution_time="30min - 2 hours"
            )
            
        # State 3: Borderline Quality = REVIEW with higher priority
        if dqs_composite >= self.DQS_REVIEW_THRESHOLD:
            self.decision_log.append("REVIEW: Borderline quality")
            
            return DecisionResult(
                action=DecisionAction.REVIEW_REQUIRED,
                priority=DecisionPriority.P3_MEDIUM,
                rationale=f"Quality borderline ({dqs_composite:.1f}%). Manual verification recommended.",
                factors=factors,
                confidence_band=confidence_band,
                responsible_party="Data Quality Team",
                next_steps=[
                    "Examine dimension breakdown",
                    "Identify weakest dimensions",
                    "Determine if remediation possible",
                    "Make proceed/reject decision"
                ],
                estimated_resolution_time="1-3 hours"
            )
            
        # State 4: Low Quality = ESCALATE
        if dqs_composite < self.DQS_REVIEW_THRESHOLD:
            self.decision_log.append("ESCALATE: Quality below acceptable threshold")
            
            return DecisionResult(
                action=DecisionAction.ESCALATE,
                priority=DecisionPriority.P2_HIGH,
                rationale=f"DQS ({dqs_composite:.1f}%) below minimum threshold ({self.DQS_REVIEW_THRESHOLD}%).",
                factors=factors,
                confidence_band=confidence_band,
                responsible_party="Data Governance Team",
                next_steps=[
                    "Investigate data source quality",
                    "Identify systemic issues",
                    "Coordinate with upstream systems",
                    "Establish remediation plan"
                ],
                estimated_resolution_time="2-6 hours"
            )
            
        # Fallback: NO_ACTION
        self.decision_log.append("NO_ACTION: Unable to determine appropriate action")
        
        return DecisionResult(
            action=DecisionAction.NO_ACTION,
            priority=DecisionPriority.P3_MEDIUM,
            rationale="System could not determine appropriate action. Manual intervention required.",
            factors=factors,
            confidence_band=confidence_band,
            responsible_party="System Administrator",
            next_steps=[
                "Review system logs",
                "Check for edge cases",
                "Manual data assessment required"
            ],
            estimated_resolution_time="Unknown"
        )
    
    def get_decision_log(self) -> List[str]:
        """Return the decision process log."""
        return self.decision_log
