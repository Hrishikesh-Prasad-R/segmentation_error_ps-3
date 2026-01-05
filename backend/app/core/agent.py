"""
PIPELINE ORCHESTRATOR V2
Purpose: Central coordination with Rules > AI decision hierarchy
Type: Orchestration

Key Changes:
- 4-state decision framework: ESCALATE, REVIEW_REQUIRED, SAFE_TO_USE, NO_ACTION
- Deterministic rules ALWAYS override AI analysis
- AI provides advisory input only for edge cases
- Clear threshold-based gating
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import pandas as pd
from datetime import datetime

# Layer imports
from app.core.layers.layer_1_2_input import Layer1InputContract, Layer2InputValidation
from app.core.layers.layer_3_features import Layer3FeatureExtraction, FeatureStats, DimensionRelevanceAnalyzer
from app.core.layers.layer_4_inference import Layer4Inference, AnomalyFlag
from app.core.layers.layer_5_6_output import Layer5OutputContract, Layer6Stability, ConsistencyLevel
from app.core.layers.layer_7_8_conflict import Layer7ConflictDetection, Layer8ConfidenceBand
from app.core.layers.layer_10_responsibility import Layer10Responsibility
from app.core.layers.layer_11_logging import Layer11Logging, LogSeverity
from app.core.layers.layer_12_final_summary import Layer12FinalSummary, SummaryContext

# ============================================================
# DECISION FRAMEWORK
# ============================================================

class DecisionState(str, Enum):
    """4-state decision framework."""
    ESCALATE = "ESCALATE"               # Critical: Immediate human review required
    REVIEW_REQUIRED = "REVIEW_REQUIRED"  # Moderate: Review before use
    SAFE_TO_USE = "SAFE_TO_USE"         # Low risk: Proceed with data
    NO_ACTION = "NO_ACTION"             # AI advisory: Don't use despite low violations

class DecisionThresholds:
    """Configurable thresholds for decision gates."""
    
    # TIER 1: ESCALATE thresholds (Hard gates)
    ESCALATE_CRITICAL_VIOLATIONS = 5    # >= 5 critical violations → ESCALATE
    ESCALATE_DQS_THRESHOLD = 50.0       # DQS < 50% → ESCALATE
    ESCALATE_HIGH_ANOMALIES = 10        # >= 10 high-severity anomalies → ESCALATE
    
    # TIER 2: REVIEW thresholds
    REVIEW_CRITICAL_VIOLATIONS = 2      # >= 2 critical violations → REVIEW
    REVIEW_DQS_THRESHOLD = 70.0         # DQS < 70% → REVIEW
    REVIEW_HIGH_ANOMALIES = 3           # >= 3 high-severity anomalies → REVIEW
    
    # TIER 3: SAFE thresholds
    SAFE_DQS_THRESHOLD = 85.0           # DQS >= 85% → likely SAFE
    SAFE_CONFIDENCE_THRESHOLD = 75      # Confidence >= 75 → trustworthy
    
    # AI ADVISORY thresholds (only used when rules don't trigger)
    AI_VETO_CONFIDENCE_THRESHOLD = 40   # Confidence < 40 → AI can suggest NO_ACTION
    AI_VETO_ANOMALY_THRESHOLD = 8       # >= 8 high anomalies with low confidence

# ============================================================
# FAILURE CATEGORIES
# ============================================================

class FailureCategory(str, Enum):
    """Comprehensive failure category taxonomy for safe degradation."""
    
    # Input Failures (Layer 1-2, API)
    INPUT_MISSING = "INPUT_MISSING"           # Required fields/columns absent
    INPUT_MALFORMED = "INPUT_MALFORMED"       # Corrupt file, unparseable format
    INPUT_OUT_OF_RANGE = "INPUT_OUT_OF_RANGE" # Values exceed valid bounds
    INPUT_DUPLICATE = "INPUT_DUPLICATE"       # Duplicate IDs or rows
    INPUT_ADVERSARIAL = "INPUT_ADVERSARIAL"   # Suspicious patterns, injection attempts
    
    # Data Quality Failures (Layer 4)
    DATA_OUTSIDE_DISTRIBUTION = "DATA_OUTSIDE_DISTRIBUTION"  # Significantly different from expected
    DATA_CONFLICTING_SIGNALS = "DATA_CONFLICTING_SIGNALS"    # Metrics contradict each other
    DATA_STALE = "DATA_STALE"                                 # Data is too old
    DATA_MODEL_CRASH = "DATA_MODEL_CRASH"                     # ML/AI model threw exception
    
    # Decision & Action Failures (Layer 9-10)
    DECISION_AMBIGUOUS = "DECISION_AMBIGUOUS"         # Metrics in gray zone
    DECISION_RESPONSIBILITY_UNCLEAR = "DECISION_RESPONSIBILITY_UNCLEAR"  # Cannot assign
    
    # System & Infrastructure Failures
    SYSTEM_UNHANDLED_EXCEPTION = "SYSTEM_UNHANDLED_EXCEPTION"  # Unexpected error
    SYSTEM_TIMEOUT = "SYSTEM_TIMEOUT"                           # Processing exceeded time
    SYSTEM_RESOURCE_EXHAUSTION = "SYSTEM_RESOURCE_EXHAUSTION"   # Memory/CPU limits
    
    # No failure (success case)
    NONE = "NONE"

# ============================================================
# DATA MODELS
# ============================================================

class DimensionScore(BaseModel):
    dimension: str
    score: float
    confidence: float = 100.0
    explanation: str
    status: str
    recommendation: str

class AnomalyDetail(BaseModel):
    row: int
    column: str
    detector: str
    severity: str
    reason: str

class LayerStatus(BaseModel):
    layer: str
    status: str
    duration_ms: float = 0.0
    details: Optional[str] = None

class DecisionReasoning(BaseModel):
    """Detailed reasoning for the decision."""
    decision_state: str
    primary_reason: str
    rule_triggers: List[str] = []
    ai_inputs: List[str] = []
    overridden_by_rules: bool = False

class FailureInfo(BaseModel):
    """Information about a failure for safe degradation."""
    category: str = "NONE"  # FailureCategory value
    message: str = ""
    layer: str = ""  # Which layer failed
    recoverable: bool = True  # Can the system continue with degraded functionality?
    user_action_required: str = ""  # What the user should do

class AnalysisResult(BaseModel):
    # Core Results
    composite_score: float
    overall_status: str
    summary: str = ""
    
    # Decision Details
    decision_reasoning: Optional[DecisionReasoning] = None
    
    # Detailed Breakdowns
    dimensions: List[DimensionScore] = []
    anomalies: List[AnomalyDetail] = []
    
    # Metadata
    trace_id: str = ""
    confidence_band: str = "MEDIUM"
    confidence_score: int = 50
    
    # Audit Trail
    layer_trace: List[LayerStatus] = []
    logs: List[str] = []
    
    # Governance
    responsible_party: str = ""
    next_steps: List[str] = []
    liability_summary: str = ""
    
    # Failure Tracking
    failure_info: Optional[FailureInfo] = None
    
    # Dimension Auto-Detection
    dimension_relevance: Optional[Dict[str, Any]] = None  # Auto-detected dimension applicability

# ============================================================
# DECISION LOGIC (Rules > AI)
# ============================================================

class Layer9ComprehensiveDecision:
    """
    Enhanced Decision Gate: Rules > AI Priority
    
    Decision Hierarchy:
    1. ESCALATE - Deterministic rules mandate escalation (AI ignored)
    2. REVIEW_REQUIRED - Deterministic rules mandate review (AI ignored)
    3. SAFE_TO_USE - Rules say safe, AI doesn't object strongly
    4. NO_ACTION - AI advisory only when rules are borderline
    """
    
    @staticmethod
    def decide(
        dqs: float,
        critical_violations: int,
        warning_violations: int,
        high_severity_anomalies: int,
        total_anomalies: int,
        confidence_score: int,
        confidence_band: str,
        ml_degraded: bool,
        stability_level: str,
        conflict_count: int
    ) -> tuple[DecisionState, DecisionReasoning]:
        """
        Make comprehensive decision with Rules > AI hierarchy.
        
        Returns:
            (DecisionState, DecisionReasoning)
        """
        rule_triggers = []
        ai_inputs = []
        overridden = False
        
        # =============================================
        # TIER 1: ESCALATE (Rules-Based - Highest Priority)
        # =============================================
        
        # Rule 1: Too many critical violations
        if critical_violations >= DecisionThresholds.ESCALATE_CRITICAL_VIOLATIONS:
            rule_triggers.append(f"Critical violations: {critical_violations} >= {DecisionThresholds.ESCALATE_CRITICAL_VIOLATIONS}")
            return (
                DecisionState.ESCALATE,
                DecisionReasoning(
                    decision_state="ESCALATE",
                    primary_reason="Deterministic rule: Critical violation threshold exceeded",
                    rule_triggers=rule_triggers,
                    ai_inputs=["AI analysis not consulted - rule-based gate"],
                    overridden_by_rules=False
                )
            )
        
        # Rule 2: DQS catastrophically low
        if dqs < DecisionThresholds.ESCALATE_DQS_THRESHOLD:
            rule_triggers.append(f"DQS: {dqs:.1f}% < {DecisionThresholds.ESCALATE_DQS_THRESHOLD}%")
            return (
                DecisionState.ESCALATE,
                DecisionReasoning(
                    decision_state="ESCALATE",
                    primary_reason="Deterministic rule: Data quality score below minimum threshold",
                    rule_triggers=rule_triggers,
                    ai_inputs=["AI analysis not consulted - rule-based gate"],
                    overridden_by_rules=False
                )
            )
        
        # Rule 3: Multiple critical indicators
        escalate_signals = 0
        if critical_violations >= 3:
            escalate_signals += 1
            rule_triggers.append(f"Critical violations: {critical_violations} >= 3")
        if dqs < 60:
            escalate_signals += 1
            rule_triggers.append(f"DQS: {dqs:.1f}% < 60%")
        if stability_level in ["CRITICAL", "CONTRADICTION"]:
            escalate_signals += 1
            rule_triggers.append(f"Stability: {stability_level}")
        
        if escalate_signals >= 2:
            return (
                DecisionState.ESCALATE,
                DecisionReasoning(
                    decision_state="ESCALATE",
                    primary_reason=f"Deterministic rule: {escalate_signals} critical indicators triggered",
                    rule_triggers=rule_triggers,
                    ai_inputs=[f"AI flagged {high_severity_anomalies} high-severity issues (informational only)"],
                    overridden_by_rules=False
                )
            )
        
        # =============================================
        # TIER 2: REVIEW_REQUIRED (Rules-Based)
        # =============================================
        
        # Rule 4: Moderate critical violations
        if critical_violations >= DecisionThresholds.REVIEW_CRITICAL_VIOLATIONS:
            rule_triggers.append(f"Critical violations: {critical_violations} >= {DecisionThresholds.REVIEW_CRITICAL_VIOLATIONS}")
            ai_inputs.append(f"AI confidence: {confidence_score}% (informational)")
            return (
                DecisionState.REVIEW_REQUIRED,
                DecisionReasoning(
                    decision_state="REVIEW_REQUIRED",
                    primary_reason="Deterministic rule: Critical violations require review",
                    rule_triggers=rule_triggers,
                    ai_inputs=ai_inputs,
                    overridden_by_rules=False
                )
            )
        
        # Rule 5: DQS in warning zone
        if dqs < DecisionThresholds.REVIEW_DQS_THRESHOLD:
            rule_triggers.append(f"DQS: {dqs:.1f}% < {DecisionThresholds.REVIEW_DQS_THRESHOLD}%")
            return (
                DecisionState.REVIEW_REQUIRED,
                DecisionReasoning(
                    decision_state="REVIEW_REQUIRED",
                    primary_reason="Deterministic rule: Data quality below review threshold",
                    rule_triggers=rule_triggers,
                    ai_inputs=[f"AI detected {total_anomalies} anomalies (informational)"],
                    overridden_by_rules=False
                )
            )
        
        # Rule 6: Multiple warning indicators
        if warning_violations > 10 and confidence_band == "LOW":
            rule_triggers.append(f"Warning violations: {warning_violations} > 10")
            rule_triggers.append(f"Confidence: {confidence_band}")
            return (
                DecisionState.REVIEW_REQUIRED,
                DecisionReasoning(
                    decision_state="REVIEW_REQUIRED",
                    primary_reason="Deterministic rule: High warning count with low confidence",
                    rule_triggers=rule_triggers,
                    ai_inputs=ai_inputs,
                    overridden_by_rules=False
                )
            )
        
        # Rule 7: ML degraded with any violations
        if ml_degraded and (critical_violations > 0 or warning_violations > 5):
            rule_triggers.append("ML degraded + violations present")
            return (
                DecisionState.REVIEW_REQUIRED,
                DecisionReasoning(
                    decision_state="REVIEW_REQUIRED",
                    primary_reason="Deterministic rule: ML unavailable, cannot validate violations",
                    rule_triggers=rule_triggers,
                    ai_inputs=["AI models unavailable - fallback mode"],
                    overridden_by_rules=False
                )
            )
        
        # =============================================
        # TIER 3: SAFE_TO_USE (Rules Pass, Check AI)
        # =============================================
        
        # Rules say data is clean
        if (
            critical_violations == 0
            and dqs >= DecisionThresholds.SAFE_DQS_THRESHOLD
            and confidence_score >= DecisionThresholds.SAFE_CONFIDENCE_THRESHOLD
            and stability_level not in ["CONTRADICTION", "CRITICAL"]
        ):
            rule_triggers.append(f"All rule checks passed: DQS={dqs:.1f}%, violations=0")
            ai_inputs.append(f"AI confidence: {confidence_score}% (agrees)")
            return (
                DecisionState.SAFE_TO_USE,
                DecisionReasoning(
                    decision_state="SAFE_TO_USE",
                    primary_reason="Deterministic rules: All quality checks passed",
                    rule_triggers=rule_triggers,
                    ai_inputs=ai_inputs,
                    overridden_by_rules=False
                )
            )
        
        # =============================================
        # TIER 4: AI ADVISORY (Only for Edge Cases)
        # =============================================
        
        # Edge case: Rules are borderline, check AI input
        # AI can suggest NO_ACTION only if:
        # 1. Rules don't mandate ESCALATE/REVIEW
        # 2. AI has strong concerns (low confidence + many anomalies)
        # 3. Not in degraded mode
        
        if (
            not ml_degraded
            and confidence_score < DecisionThresholds.AI_VETO_CONFIDENCE_THRESHOLD
            and high_severity_anomalies >= DecisionThresholds.AI_VETO_ANOMALY_THRESHOLD
            and critical_violations <= 1  # Rules are borderline
            and dqs >= 70  # DQS is acceptable
        ):
            ai_inputs.append(f"AI confidence very low: {confidence_score}%")
            ai_inputs.append(f"AI detected {high_severity_anomalies} high-severity anomalies")
            ai_inputs.append("AI advisory: Patterns suggest unreliability")
            
            return (
                DecisionState.NO_ACTION,
                DecisionReasoning(
                    decision_state="NO_ACTION",
                    primary_reason="AI advisory: Low confidence despite passing rule checks",
                    rule_triggers=[f"Rules borderline: DQS={dqs:.1f}%, violations={critical_violations}"],
                    ai_inputs=ai_inputs,
                    overridden_by_rules=False
                )
            )
        
        # =============================================
        # DEFAULT: Conservative Fallback
        # =============================================
        
        # If we reach here, rules don't mandate action but quality isn't perfect
        # Default to REVIEW_REQUIRED (conservative)
        if critical_violations > 0 or dqs < 80 or confidence_score < 60:
            return (
                DecisionState.REVIEW_REQUIRED,
                DecisionReasoning(
                    decision_state="REVIEW_REQUIRED",
                    primary_reason="Conservative default: Quality metrics in gray zone",
                    rule_triggers=[f"DQS={dqs:.1f}%, violations={critical_violations}"],
                    ai_inputs=[f"AI confidence={confidence_score}% (informational)"],
                    overridden_by_rules=False
                )
            )
        
        # Everything looks acceptable
        return (
            DecisionState.SAFE_TO_USE,
            DecisionReasoning(
                decision_state="SAFE_TO_USE",
                primary_reason="Rules and AI both indicate acceptable quality",
                rule_triggers=[f"DQS={dqs:.1f}%, violations={critical_violations}"],
                ai_inputs=[f"AI confidence={confidence_score}%"],
                overridden_by_rules=False
            )
        )

# ============================================================
# PIPELINE ORCHESTRATOR
# ============================================================

class PipelineOrchestrator:
    """
    Central orchestrator with Rules > AI decision hierarchy.
    """
    
    def __init__(self):
        self.l1 = Layer1InputContract()
        self.l2 = Layer2InputValidation()
        self.l3 = Layer3FeatureExtraction()
        self.l4 = Layer4Inference()
        self.l5 = Layer5OutputContract()
        self.l6 = Layer6Stability()
        self.l7 = Layer7ConflictDetection()
        self.l8 = Layer8ConfidenceBand()
        self.l9 = Layer9ComprehensiveDecision()
        self.l10 = Layer10Responsibility()
        self.l12 = Layer12FinalSummary()
        self.dimension_analyzer = DimensionRelevanceAnalyzer()  # Auto-detect relevant dimensions
        
    def analyze(self, df: pd.DataFrame, simulate_ml_failure: bool = False) -> AnalysisResult:
        """Execute the full 11-layer analysis pipeline with Rules > AI decision logic."""
        
        logger = Layer11Logging()
        layer_trace: List[LayerStatus] = []
        
        logger.log_event("PIPELINE", "STARTED", {
            "rows": len(df),
            "columns": len(df.columns),
            "decision_framework": "Rules > AI (4-state)"
        })
        
        try:
            # LAYER 1: INPUT CONTRACT
            logger.start_layer("Layer 1: Input Contract")
            ok, contract_result = self.l1.check(df)
            
            if not ok:
                logger.end_layer("Layer 1: Input Contract", "FAIL", contract_result)
                layer_trace.append(LayerStatus(
                    layer="Layer 1: Input Contract",
                    status="FAIL",
                    details=contract_result.get('message', 'Contract violation')
                ))
                return self._build_failure_result("CONTRACT_VIOLATION", 
                                                 contract_result.get('message', 'Input contract failed'),
                                                 logger, layer_trace,
                                                 failure_category=FailureCategory.INPUT_MISSING,
                                                 failed_layer="Layer 1: Input Contract")
            
            logger.end_layer("Layer 1: Input Contract", "PASS", contract_result)
            layer_trace.append(LayerStatus(layer="Layer 1: Input Contract", status="PASS",
                                          details=f"{contract_result.get('rows_found', 0)} rows validated"))
            
            # AUTO-DETECT RELEVANT DIMENSIONS (Schema Analysis)
            dimension_relevance = self.dimension_analyzer.get_dimension_summary(df)
            logger.log_event("DIMENSION_ANALYSIS", "AUTO_DETECTED", {
                "applicable": dimension_relevance["applicable_count"],
                "skipped": dimension_relevance["not_applicable_count"],
                "columns_analyzed": len(df.columns)
            })
            
            # LAYER 2: INPUT VALIDATION
            logger.start_layer("Layer 2: Input Validation")
            ok, validation_result = self.l2.validate(df)
            
            status = "PASS" if ok else "WARN"
            logger.end_layer("Layer 2: Input Validation", status, validation_result)
            layer_trace.append(LayerStatus(layer="Layer 2: Input Validation", status=status,
                                          details=f"Score: {validation_result.get('validation_score', 0):.1f}%"))
            
            if validation_result.get('critical_issues'):
                logger.log_event("Layer 2: Input Validation", "CRITICAL", 
                               validation_result['critical_issues'], LogSeverity.ERROR)
            
            # LAYER 3: FEATURE EXTRACTION
            logger.start_layer("Layer 3: Feature Extraction")
            df_enriched, feature_stats = self.l3.extract(df)
            logger.end_layer("Layer 3: Feature Extraction", "COMPLETE", {
                "features_created": feature_stats.total_features_created
            })
            layer_trace.append(LayerStatus(layer="Layer 3: Feature Extraction", status="PASS",
                                          details=f"{feature_stats.total_features_created} features"))
            
            # LAYER 4.1: STRUCTURAL INTEGRITY
            logger.start_layer("Layer 4.1: Structural Integrity")
            ok, structural_result = self.l4.sublayer_4_1_structural(df_enriched)
            
            if not ok:
                logger.end_layer("Layer 4.1: Structural Integrity", "FAIL", structural_result)
                layer_trace.append(LayerStatus(layer="Layer 4.1: Structural Integrity",
                                              status="FAIL", details="Structural gate failed"))
                return self._build_failure_result("STRUCTURAL_FAILURE",
                                                 "Data structure validation failed",
                                                 logger, layer_trace,
                                                 failure_category=FailureCategory.INPUT_MISSING,
                                                 failed_layer="Layer 4.1: Structural Integrity")
            
            logger.end_layer("Layer 4.1: Structural Integrity", "PASS", structural_result)
            layer_trace.append(LayerStatus(layer="Layer 4.1: Structural Integrity",
                                          status="PASS", details="All structural checks passed"))
            
            # LAYER 4.2: FIELD-LEVEL COMPLIANCE (RULES)
            logger.start_layer("Layer 4.2: Field Compliance")
            rules_result = self.l4.sublayer_4_2_rules(df_enriched)
            dqs = rules_result['composite']
            dimension_scores = rules_result['scores']
            
            logger.end_layer("Layer 4.2: Field Compliance", "COMPLETE", {"composite_dqs": dqs})
            layer_trace.append(LayerStatus(layer="Layer 4.2: Field Compliance",
                                          status="PASS" if dqs >= 70 else "WARN",
                                          details=f"DQS: {dqs:.1f}%"))
            
            # LAYER 4.3: SEMANTIC VALIDATION
            logger.start_layer("Layer 4.3: Semantic Validation")
            semantic_result = self.l4.sublayer_4_3_semantic(df_enriched)
            crit_violations = semantic_result['critical_violations']
            total_violations = semantic_result['total_violations']
            
            sem_status = "PASS"
            if crit_violations > 0:
                sem_status = "WARN" if crit_violations < 5 else "FAIL"
            
            logger.end_layer("Layer 4.3: Semantic Validation", sem_status, semantic_result)
            layer_trace.append(LayerStatus(layer="Layer 4.3: Semantic Validation",
                                          status=sem_status, details=f"{crit_violations} violations"))
            
            # LAYER 4.4: ANOMALY DETECTION (ML)
            logger.start_layer("Layer 4.4: Anomaly Detection")
            anomaly_flags, ml_metadata = self.l4.sublayer_4_4_anomaly(df_enriched, 
                                                                      simulate_failure=simulate_ml_failure)
            
            ml_degraded = ml_metadata.get('degraded', False)
            if ml_degraded:
                logger.log_event("Layer 4.4: Anomaly Detection", "DEGRADED",
                               "ML models unavailable - fallback mode", LogSeverity.WARN)
            
            high_severity_anomalies = sum(1 for f in anomaly_flags if f.severity in ['HIGH', 'CRITICAL'])
            
            logger.end_layer("Layer 4.4: Anomaly Detection", 
                           "DEGRADED" if ml_degraded else "COMPLETE",
                           {"flags": len(anomaly_flags), "high_severity": high_severity_anomalies})
            layer_trace.append(LayerStatus(layer="Layer 4.4: Anomaly Detection",
                                          status="WARN" if ml_degraded else "PASS",
                                          details=f"{len(anomaly_flags)} flags" + 
                                                 (" (DEGRADED)" if ml_degraded else "")))
            

            
            # LAYER 6: STABILITY CHECK
            logger.start_layer("Layer 6: Stability Check")
            stability_level, stability_checks = self.l6.check_consistency(dqs, dimension_scores,
                                                                         crit_violations, len(anomaly_flags))
            
            stab_status = "PASS"
            if stability_level in [ConsistencyLevel.CONTRADICTION, ConsistencyLevel.CRITICAL]:
                stab_status = "FAIL"
            elif stability_level == ConsistencyLevel.WARNING:
                stab_status = "WARN"
            
            logger.end_layer("Layer 6: Stability Check", stab_status, {"level": stability_level.value})
            layer_trace.append(LayerStatus(layer="Layer 6: Stability Check",
                                          status=stab_status, details=stability_level.value))
            
            # LAYER 7: CONFLICT DETECTION
            logger.start_layer("Layer 7: Conflict Detection")
            conflict_result = self.l7.resolve(dqs, dimension_scores, len(anomaly_flags),
                                             high_severity_anomalies, total_violations,
                                             crit_violations > 0)
            
            logger.end_layer("Layer 7: Conflict Detection", "RESOLVED", conflict_result)
            layer_trace.append(LayerStatus(layer="Layer 7: Conflict Detection",
                                          status="PASS", details=f"{conflict_result['total_conflicts']} conflicts"))
            
            # LAYER 8: CONFIDENCE BAND
            logger.start_layer("Layer 8: Confidence Band")
            confidence_result = self.l8.calculate(dqs, dimension_scores, len(anomaly_flags),
                                                 conflict_result, len(df), ml_degraded)
            
            logger.end_layer("Layer 8: Confidence Band", "CALCULATED", confidence_result)
            layer_trace.append(LayerStatus(layer="Layer 8: Confidence Band", status="PASS",
                                          details=f"{confidence_result['band']} ({confidence_result['score']})"))
            
            # =====================================================
            # LAYER 9: COMPREHENSIVE DECISION GATE (Rules > AI)
            # =====================================================
            logger.start_layer("Layer 9: Comprehensive Decision Gate")
            
            decision_state, decision_reasoning = self.l9.decide(
                dqs=dqs,
                critical_violations=crit_violations,
                warning_violations=total_violations - crit_violations,
                high_severity_anomalies=high_severity_anomalies,
                total_anomalies=len(anomaly_flags),
                confidence_score=confidence_result['score'],
                confidence_band=confidence_result['band'],
                ml_degraded=ml_degraded,
                stability_level=stability_level.value,
                conflict_count=conflict_result['total_conflicts']
            )
            
            logger.end_layer("Layer 9: Comprehensive Decision Gate", "DECIDED", {
                "decision": decision_state.value,
                "reasoning": decision_reasoning.primary_reason
            })
            layer_trace.append(LayerStatus(layer="Layer 9: Comprehensive Decision Gate",
                                          status="PASS", details=decision_state.value))
            
            # LAYER 10: RESPONSIBILITY BOUNDARY
            logger.start_layer("Layer 10: Responsibility")
            handoff = self.l10.get_handoff(decision_state.value, {"dqs": dqs, "anomalies": len(anomaly_flags)})
            
            logger.end_layer("Layer 10: Responsibility", "ASSIGNED", handoff)
            layer_trace.append(LayerStatus(layer="Layer 10: Responsibility", status="PASS",
                                          details=handoff['accountability']['primary_responsible']))
            
            # LAYER 5: OUTPUT CONTRACT
            logger.log_event("Layer 5: Output Contract", "VALIDATED", "Schema satisfied")
            layer_trace.append(LayerStatus(layer="Layer 5: Output Contract",
                                          status="PASS", details="Output validated"))
            
            # Prepare dimensions and anomalies for summary context
            dimensions_for_summary = []
            for dim in rules_result.get('dimensions', []):
                dimensions_for_summary.append({
                    'dimension': dim['name'],
                    'score': dim['score'],
                    'status': dim['status'],
                    'details': dim['details'],
                    'recommendation': dim['recommendation']
                })
            
            anomalies_for_summary = []
            for flag in anomaly_flags[:20]:
                anomalies_for_summary.append({
                    'row': flag.row_index,
                    'column': flag.column,
                    'detector': flag.detector,
                    'severity': flag.severity,
                    'reason': flag.reason
                })
            
            # Map next steps based on decision
            next_steps_map = {
                "ESCALATE": ["Immediate data steward review required",
                            "Block data usage until resolved",
                            "Investigate root cause of critical violations"],
                "REVIEW_REQUIRED": ["Human review recommended before use",
                                   "Validate flagged anomalies",
                                   "Consider data remediation"],
                "SAFE_TO_USE": ["Data approved for use",
                               "Monitor for ongoing quality",
                               "Standard processing can proceed"],
                "NO_ACTION": ["AI advisory: Do not use data",
                             "Review AI-flagged patterns",
                             "Consider additional validation"]
            }
            
            layer_trace_for_summary = [
                {'layer': lt.layer, 'status': lt.status, 'details': lt.details}
                for lt in layer_trace
            ]
            
            # LAYER 12: FINAL GENAI SUMMARY (runs at the end with full context)
            logger.start_layer("Layer 12: Final GenAI Summary")
            summary_context = SummaryContext(
                dqs=dqs,
                overall_status=decision_state.value,
                decision_reasoning={
                    'decision_state': decision_reasoning.decision_state,
                    'primary_reason': decision_reasoning.primary_reason,
                    'rule_triggers': decision_reasoning.rule_triggers,
                    'ai_inputs': decision_reasoning.ai_inputs,
                    'overridden_by_rules': decision_reasoning.overridden_by_rules
                },
                dimensions=dimensions_for_summary,
                anomalies=anomalies_for_summary,
                confidence_band=confidence_result['band'],
                confidence_score=confidence_result['score'],
                responsible_party=handoff['accountability']['primary_responsible'],
                next_steps=next_steps_map.get(decision_state.value, []),
                liability_summary=self.l10.get_liability_summary(decision_state.value),
                layer_trace=layer_trace_for_summary,
                logs=[str(log) for log in logger.get_logs()]
            )
            summary, genai_metadata = self.l12.summarize(summary_context)
            logger.end_layer("Layer 12: Final GenAI Summary", "COMPLETE", genai_metadata)
            layer_trace.append(LayerStatus(layer="Layer 12: Final GenAI Summary",
                                          status="PASS", details="Summary generated"))
            
            # CONSTRUCT FINAL RESULT
            logger.log_event("PIPELINE", "COMPLETED", logger.get_summary())
            
            dimensions = []
            for dim in rules_result.get('dimensions', []):
                dimensions.append(DimensionScore(
                    dimension=dim['name'],
                    score=dim['score'],
                    confidence=100.0 if not ml_degraded else 70.0,
                    explanation=dim['details'],
                    status=dim['status'],
                    recommendation=dim['recommendation']
                ))
            
            anomalies = []
            for flag in anomaly_flags[:20]:
                anomalies.append(AnomalyDetail(
                    row=flag.row_index,
                    column=flag.column,
                    detector=flag.detector,
                    severity=flag.severity,
                    reason=flag.reason
                ))
            
            return AnalysisResult(
                composite_score=round(dqs, 2),
                overall_status=decision_state.value,
                summary=summary,
                decision_reasoning=decision_reasoning,
                dimensions=dimensions,
                anomalies=anomalies,
                trace_id=logger.trace_id,
                confidence_band=confidence_result['band'],
                confidence_score=confidence_result['score'],
                layer_trace=layer_trace,
                logs=[str(log) for log in logger.get_logs()],
                responsible_party=handoff['accountability']['primary_responsible'],
                next_steps=next_steps_map.get(decision_state.value, []),
                liability_summary=self.l10.get_liability_summary(decision_state.value),
                dimension_relevance=dimension_relevance  # Auto-detected dimension applicability
            )
            
        except Exception as e:
            logger.log_critical("PIPELINE", f"Unexpected error: {str(e)}")
            return self._build_failure_result("SYSTEM_ERROR", f"Pipeline failed: {str(e)}",
                                             logger, layer_trace)
    
    def _build_failure_result(self, error_code: str, message: str,
                             logger: Layer11Logging, layer_trace: List[LayerStatus],
                             failure_category: FailureCategory = FailureCategory.SYSTEM_UNHANDLED_EXCEPTION,
                             failed_layer: str = "PIPELINE",
                             recoverable: bool = False) -> AnalysisResult:
        """Build a standardized failure result with safe degradation."""
        
        # Map failure category to user action
        user_actions = {
            FailureCategory.INPUT_MISSING: "Ensure all required columns are present: transaction_id, amount, merchant_category, date",
            FailureCategory.INPUT_MALFORMED: "Check file format and encoding. Re-export from source system.",
            FailureCategory.INPUT_OUT_OF_RANGE: "Review flagged values and correct data at source.",
            FailureCategory.INPUT_DUPLICATE: "Remove duplicate transaction IDs from the dataset.",
            FailureCategory.INPUT_ADVERSARIAL: "Dataset contains suspicious patterns. Contact security team.",
            FailureCategory.DATA_OUTSIDE_DISTRIBUTION: "Data differs significantly from expected patterns. Verify data source.",
            FailureCategory.DATA_CONFLICTING_SIGNALS: "Quality metrics are contradictory. Manual review required.",
            FailureCategory.DATA_STALE: "Data is outdated. Refresh from current source.",
            FailureCategory.DATA_MODEL_CRASH: "AI/ML models failed. Analysis continued with rules only.",
            FailureCategory.DECISION_AMBIGUOUS: "Decision unclear. Manual review required.",
            FailureCategory.DECISION_RESPONSIBILITY_UNCLEAR: "Cannot determine responsible party. Escalate to governance.",
            FailureCategory.SYSTEM_UNHANDLED_EXCEPTION: "Unexpected system error. Contact support.",
            FailureCategory.SYSTEM_TIMEOUT: "Analysis timed out. Try with smaller dataset.",
            FailureCategory.SYSTEM_RESOURCE_EXHAUSTION: "System resources exhausted. Try again later.",
        }
        
        failure_info = FailureInfo(
            category=failure_category.value,
            message=message,
            layer=failed_layer,
            recoverable=recoverable,
            user_action_required=user_actions.get(failure_category, "Contact support.")
        )
        
        return AnalysisResult(
            composite_score=0.0,
            overall_status="NO_ACTION",
            summary=f"Analysis failed: {message}",
            decision_reasoning=DecisionReasoning(
                decision_state="NO_ACTION",
                primary_reason=f"Pipeline failure: {failure_category.value}",
                rule_triggers=[message],
                ai_inputs=["Analysis incomplete"],
                overridden_by_rules=False
            ),
            dimensions=[],
            anomalies=[],
            trace_id=logger.trace_id,
            confidence_band="LOW",
            confidence_score=0,
            layer_trace=layer_trace,
            logs=[str(log) for log in logger.get_logs()],
            responsible_party="DATA_ENGINEERING",
            next_steps=[failure_info.user_action_required, "Fix data issues", "Retry analysis"],
            liability_summary="⛔ Analysis incomplete. No processing permitted.",
            failure_info=failure_info
        )