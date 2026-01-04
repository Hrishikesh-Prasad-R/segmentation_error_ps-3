"""
LAYERS 7 & 8: CONFLICT DETECTION & CONFIDENCE BAND
Purpose: Resolve contradictory signals and calculate decision confidence.
Type: 100% Deterministic

Features:
- Multi-type conflict resolution (Rules vs ML, Score vs Violations, etc.)
- Priority-based resolution (Rules > ML > Heuristics)
- Confidence scoring with detailed deductions
- Band classification (HIGH/MEDIUM/LOW)
"""
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ConflictType(Enum):
    RULES_VS_ML = "RULES_VS_ML"
    SCORE_VS_VIOLATIONS = "SCORE_VS_VIOLATIONS"
    POSITIVE_NEGATIVE_SIGNAL = "POSITIVE_NEGATIVE_SIGNAL"
    DIMENSION_DISAGREEMENT = "DIMENSION_DISAGREEMENT"
    TEMPORAL_INCONSISTENCY = "TEMPORAL_INCONSISTENCY"

class ConflictResolution(Enum):
    RULES_AUTHORITY = "RULES_AUTHORITY"
    ML_ADVISORY = "ML_ADVISORY"
    ESCALATE = "ESCALATE"
    OVERRIDE = "OVERRIDE"
    MERGE = "MERGE"

@dataclass
class ConflictRecord:
    """Record of a detected conflict and its resolution."""
    conflict_type: ConflictType
    resolution: ConflictResolution
    priority_source: str  # Which source took priority
    overridden_source: str  # Which source was overridden
    rationale: str
    impact_on_decision: str
    
    def to_dict(self) -> Dict:
        return {
            "type": self.conflict_type.value,
            "resolution": self.resolution.value,
            "priority_source": self.priority_source,
            "overridden_source": self.overridden_source,
            "rationale": self.rationale,
            "impact": self.impact_on_decision
        }

class Layer7ConflictDetection:
    """
    LAYER 7: CONFLICT DETECTION LAYER
    Purpose: Detect and resolve contradictory signals.
    Type: 100% Deterministic
    
    Resolution Priority:
    1. Critical Rules (immutable)
    2. Business Rules
    3. Statistical Analysis
    4. ML Predictions (advisory only)
    """
    
    def __init__(self):
        self.conflicts: List[ConflictRecord] = []
        
    def resolve(self, 
               rule_dqs: float,
               dimension_scores: Dict[str, float],
               ml_anomaly_count: int,
               ml_high_severity_count: int,
               semantic_violations: int,
               semantic_critical: bool) -> Dict[str, Any]:
        """
        Detect and resolve all conflicts.
        Returns: Resolution summary
        """
        self.conflicts = []
        
        # =====================================================
        # CONFLICT TYPE A: Rules vs ML Disagreement
        # =====================================================
        # High DQS from rules but ML flags many anomalies
        if rule_dqs > 85 and ml_anomaly_count > 10:
            self.conflicts.append(ConflictRecord(
                conflict_type=ConflictType.RULES_VS_ML,
                resolution=ConflictResolution.RULES_AUTHORITY,
                priority_source="Rule-based DQS",
                overridden_source="ML Anomaly Detection",
                rationale=f"Rules scored {rule_dqs:.1f}% while ML flagged {ml_anomaly_count} anomalies. Per 'Rules > ML' principle, rules take authority.",
                impact_on_decision="ML flags treated as ADVISORY only. Will appear in report but not affect primary decision."
            ))
            
        # Low DQS from rules but ML found no issues
        if rule_dqs < 70 and ml_anomaly_count == 0:
            self.conflicts.append(ConflictRecord(
                conflict_type=ConflictType.RULES_VS_ML,
                resolution=ConflictResolution.RULES_AUTHORITY,
                priority_source="Rule-based DQS",
                overridden_source="ML (no anomalies)",
                rationale=f"Rules scored low ({rule_dqs:.1f}%) despite ML finding no anomalies. Rules identify structural issues ML may miss.",
                impact_on_decision="Low score stands. ML absence of anomalies does not override rule-based concerns."
            ))
            
        # =====================================================
        # CONFLICT TYPE B: Score vs Violations Disagreement
        # =====================================================
        if rule_dqs > 80 and semantic_violations > 0:
            if semantic_critical:
                self.conflicts.append(ConflictRecord(
                    conflict_type=ConflictType.SCORE_VS_VIOLATIONS,
                    resolution=ConflictResolution.ESCALATE,
                    priority_source="Semantic Violations",
                    overridden_source="Composite Score",
                    rationale=f"Score is {rule_dqs:.1f}% but {semantic_violations} semantic violations exist (critical: True)",
                    impact_on_decision="ESCALATE decision forced. Critical violations override positive score."
                ))
            else:
                self.conflicts.append(ConflictRecord(
                    conflict_type=ConflictType.SCORE_VS_VIOLATIONS,
                    resolution=ConflictResolution.MERGE,
                    priority_source="Both sources",
                    overridden_source="None",
                    rationale=f"Score is {rule_dqs:.1f}% with {semantic_violations} non-critical violations",
                    impact_on_decision="REVIEW_REQUIRED. Score indicates quality but violations need human attention."
                ))
                
        # =====================================================
        # CONFLICT TYPE C: Dimension Disagreement
        # =====================================================
        if dimension_scores:
            scores = list(dimension_scores.values())
            if scores:
                score_range = max(scores) - min(scores)
                
                if score_range > 40:
                    worst_dim = min(dimension_scores, key=dimension_scores.get)
                    best_dim = max(dimension_scores, key=dimension_scores.get)
                    
                    self.conflicts.append(ConflictRecord(
                        conflict_type=ConflictType.DIMENSION_DISAGREEMENT,
                        resolution=ConflictResolution.ML_ADVISORY,
                        priority_source=f"Lowest dimension ({worst_dim})",
                        overridden_source=f"Composite average",
                        rationale=f"{worst_dim}: {dimension_scores[worst_dim]:.1f}% vs {best_dim}: {dimension_scores[best_dim]:.1f}%",
                        impact_on_decision="Individual dimension failure may trigger review even if composite is acceptable."
                    ))
                    
        # =====================================================
        # CONFLICT TYPE D: High ML Severity Override
        # =====================================================
        if ml_high_severity_count > 5 and rule_dqs > 70:
            self.conflicts.append(ConflictRecord(
                conflict_type=ConflictType.POSITIVE_NEGATIVE_SIGNAL,
                resolution=ConflictResolution.MERGE,
                priority_source="ML High-Severity Flags",
                overridden_source="None (merged)",
                rationale=f"{ml_high_severity_count} high-severity ML flags despite acceptable DQS ({rule_dqs:.1f}%)",
                impact_on_decision="ML flags will be highlighted in output. May trigger review if threshold exceeded."
            ))
            
        # Build summary
        return {
            "total_conflicts": len(self.conflicts),
            "conflicts": [c.to_dict() for c in self.conflicts],
            "resolution_summary": self._get_resolution_summary(),
            "escalation_required": any(c.resolution == ConflictResolution.ESCALATE for c in self.conflicts),
            "rules_authority_applied": any(c.resolution == ConflictResolution.RULES_AUTHORITY for c in self.conflicts)
        }
    
    def _get_resolution_summary(self) -> str:
        """Generate human-readable summary of resolutions."""
        if not self.conflicts:
            return "No conflicts detected. All signals align."
            
        escalations = sum(1 for c in self.conflicts if c.resolution == ConflictResolution.ESCALATE)
        rules_auth = sum(1 for c in self.conflicts if c.resolution == ConflictResolution.RULES_AUTHORITY)
        
        if escalations > 0:
            return f"ESCALATION REQUIRED: {escalations} conflict(s) triggered mandatory escalation."
        elif rules_auth > 0:
            return f"Rules Authority applied for {rules_auth} conflict(s). ML signals treated as advisory."
        else:
            return f"{len(self.conflicts)} conflict(s) resolved via standard priority."


class ConfidenceBand(Enum):
    HIGH = "HIGH"      # 80-100 points
    MEDIUM = "MEDIUM"  # 50-79 points
    LOW = "LOW"        # 0-49 points

@dataclass
class ConfidenceDeduction:
    """Record of a confidence score deduction."""
    reason: str
    points_deducted: int
    category: str
    
class Layer8ConfidenceBand:
    """
    LAYER 8: CONFIDENCE BAND LAYER
    Purpose: Calculate and classify decision confidence.
    Type: 100% Deterministic
    
    Starts at 100 points, deducts for:
    - Borderline scores
    - High anomaly counts
    - Unresolved conflicts
    - Low data volume
    - Missing dimensions
    """
    
    def __init__(self):
        self.deductions: List[ConfidenceDeduction] = []
        
    def calculate(self, 
                 dqs: float,
                 dimension_scores: Dict[str, float],
                 anomaly_count: int,
                 conflict_result: Dict[str, Any],
                 row_count: int = 0,
                 ml_degraded: bool = False) -> Dict[str, Any]:
        """
        Calculate confidence score and band.
        Returns: Confidence result with breakdown.
        """
        self.deductions = []
        points = 100
        
        # =====================================================
        # DEDUCTION: Borderline DQS (40-85 is "gray zone")
        # =====================================================
        if 40 <= dqs <= 85:
            deduction = int((85 - dqs) / 2)  # Up to 22.5 points
            points -= deduction
            self.deductions.append(ConfidenceDeduction(
                reason=f"DQS in borderline zone ({dqs:.1f}%)",
                points_deducted=deduction,
                category="score_quality"
            ))
            
        # =====================================================
        # DEDUCTION: Low DQS
        # =====================================================
        if dqs < 50:
            points -= 25
            self.deductions.append(ConfidenceDeduction(
                reason=f"DQS critically low ({dqs:.1f}%)",
                points_deducted=25,
                category="score_quality"
            ))
            
        # =====================================================
        # DEDUCTION: High Anomaly Count
        # =====================================================
        if anomaly_count > 20:
            deduction = min(15, anomaly_count // 5)
            points -= deduction
            self.deductions.append(ConfidenceDeduction(
                reason=f"High anomaly count ({anomaly_count})",
                points_deducted=deduction,
                category="ml_signals"
            ))
        elif anomaly_count > 5:
            points -= 5
            self.deductions.append(ConfidenceDeduction(
                reason=f"Moderate anomaly count ({anomaly_count})",
                points_deducted=5,
                category="ml_signals"
            ))
            
        # =====================================================
        # DEDUCTION: Unresolved Conflicts
        # =====================================================
        conflicts = conflict_result.get('conflicts', [])
        if len(conflicts) > 0:
            conflict_deduction = min(25, len(conflicts) * 10)
            points -= conflict_deduction
            self.deductions.append(ConfidenceDeduction(
                reason=f"{len(conflicts)} signal conflict(s) detected",
                points_deducted=conflict_deduction,
                category="conflicts"
            ))
            
        if conflict_result.get('escalation_required'):
            points -= 15
            self.deductions.append(ConfidenceDeduction(
                reason="Escalation was required",
                points_deducted=15,
                category="conflicts"
            ))
            
        # =====================================================
        # DEDUCTION: Low Data Volume
        # =====================================================
        if row_count > 0 and row_count < 10:
            points -= 10
            self.deductions.append(ConfidenceDeduction(
                reason=f"Low data volume ({row_count} rows)",
                points_deducted=10,
                category="data_quality"
            ))
            
        # =====================================================
        # DEDUCTION: Dimension Score Variance
        # =====================================================
        if dimension_scores:
            scores = list(dimension_scores.values())
            if scores:
                variance = max(scores) - min(scores)
                if variance > 30:
                    deduction = int(variance / 10)
                    points -= deduction
                    self.deductions.append(ConfidenceDeduction(
                        reason=f"High dimension variance ({variance:.1f} points)",
                        points_deducted=deduction,
                        category="dimension_quality"
                    ))
                    
                # Individual poor dimensions
                for dim, score in dimension_scores.items():
                    if score < 60:
                        points -= 5
                        self.deductions.append(ConfidenceDeduction(
                            reason=f"{dim} critically low ({score:.1f}%)",
                            points_deducted=5,
                            category="dimension_quality"
                        ))
                        
        # =====================================================
        # DEDUCTION: ML Degradation
        # =====================================================
        if ml_degraded:
            points -= 10
            self.deductions.append(ConfidenceDeduction(
                reason="ML models degraded (running in fallback mode)",
                points_deducted=10,
                category="system_health"
            ))
            
        # Ensure score stays within bounds
        points = max(0, min(100, points))
        
        # Determine band
        if points >= 80:
            band = ConfidenceBand.HIGH
        elif points >= 50:
            band = ConfidenceBand.MEDIUM
        else:
            band = ConfidenceBand.LOW
            
        return {
            "score": points,
            "band": band.value,
            "max_score": 100,
            "deductions": [
                {
                    "reason": d.reason,
                    "points": d.points_deducted,
                    "category": d.category
                }
                for d in self.deductions
            ],
            "deduction_summary": {
                category: sum(d.points_deducted for d in self.deductions if d.category == category)
                for category in set(d.category for d in self.deductions)
            }
        }
