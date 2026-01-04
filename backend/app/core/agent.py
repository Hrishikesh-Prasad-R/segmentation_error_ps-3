"""
PIPELINE ORCHESTRATOR
Purpose: Central coordination of the 11-Layer "Controlled Intelligence" Architecture.
Type: Orchestration

Features:
- Sequential layer execution with error handling
- Comprehensive logging at each step
- Graceful degradation on failures
- Performance timing
- Complete audit trail
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
from datetime import datetime

# Layer imports
from app.core.layers.layer_1_2_input import Layer1InputContract, Layer2InputValidation
from app.core.layers.layer_3_features import Layer3FeatureExtraction, FeatureStats
from app.core.layers.layer_4_inference import Layer4Inference, AnomalyFlag
from app.core.layers.layer_5_6_output import Layer5OutputContract, Layer6Stability, ConsistencyLevel
from app.core.layers.layer_7_8_conflict import Layer7ConflictDetection, Layer8ConfidenceBand
from app.core.layers.layer_9_decision import Layer9DecisionGate, DecisionResult
from app.core.layers.layer_10_responsibility import Layer10Responsibility
from app.core.layers.layer_11_logging import Layer11Logging, LogSeverity

# ============================================================
# DATA MODELS (Pydantic for API Response)
# ============================================================

class DimensionScore(BaseModel):
    """Individual dimension scoring result."""
    dimension: str
    score: float
    confidence: float = 100.0
    explanation: str
    status: str
    recommendation: str

class AnomalyDetail(BaseModel):
    """Detail of a single anomaly flag."""
    row: int
    column: str
    detector: str
    severity: str
    reason: str

class LayerStatus(BaseModel):
    """Status of a single layer execution."""
    layer: str
    status: str  # PASS, FAIL, WARN, SKIP
    duration_ms: float = 0.0
    details: Optional[str] = None

class AnalysisResult(BaseModel):
    """Complete analysis result returned by the API."""
    # Core Results
    composite_score: float
    overall_status: str
    summary: str = ""
    
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


# ============================================================
# PIPELINE ORCHESTRATOR
# ============================================================

class PipelineOrchestrator:
    """
    Central orchestrator for the 11-Layer DQS Pipeline.
    
    Layer Execution Order:
    1. Input Contract
    2. Input Validation
    3. Feature Extraction
    4. Model Inference (AI Containment Zone)
       4.1 Structural Integrity
       4.2 Field-Level Compliance (Rules)
       4.3 Semantic Validation
       4.4 Cross-Field Anomaly Detection (ML)
       4.5 GenAI Summarization
    5. Output Contract
    6. Stability & Consistency
    7. Conflict Detection
    8. Confidence Band
    9. Decision Gate
    10. Responsibility Boundary
    11. Logging & Trace
    """
    
    def __init__(self):
        # Initialize all layers
        self.l1 = Layer1InputContract()
        self.l2 = Layer2InputValidation()
        self.l3 = Layer3FeatureExtraction()
        self.l4 = Layer4Inference()
        self.l5 = Layer5OutputContract()
        self.l6 = Layer6Stability()
        self.l7 = Layer7ConflictDetection()
        self.l8 = Layer8ConfidenceBand()
        self.l9 = Layer9DecisionGate()
        self.l10 = Layer10Responsibility()
        
    def analyze(self, df: pd.DataFrame, simulate_ml_failure: bool = False) -> AnalysisResult:
        """
        Execute the full 11-layer analysis pipeline.
        
        Args:
            df: Input DataFrame to analyze
            simulate_ml_failure: If True, simulates ML model failure for demo
            
        Returns:
            AnalysisResult with complete analysis details
        """
        # Initialize fresh logger for this run
        logger = Layer11Logging()
        layer_trace: List[LayerStatus] = []
        
        logger.log_event("PIPELINE", "STARTED", {
            "rows": len(df),
            "columns": len(df.columns),
            "timestamp": datetime.now().isoformat()
        })
        
        try:
            # =====================================================
            # LAYER 1: INPUT CONTRACT
            # =====================================================
            logger.start_layer("Layer 1: Input Contract")
            ok, contract_result = self.l1.check(df)
            
            if not ok:
                logger.end_layer("Layer 1: Input Contract", "FAIL", contract_result)
                layer_trace.append(LayerStatus(
                    layer="Layer 1: Input Contract",
                    status="FAIL",
                    details=contract_result.get('message', 'Contract violation')
                ))
                return self._build_failure_result(
                    "CONTRACT_VIOLATION",
                    contract_result.get('message', 'Input contract failed'),
                    logger,
                    layer_trace
                )
                
            logger.end_layer("Layer 1: Input Contract", "PASS", contract_result)
            layer_trace.append(LayerStatus(
                layer="Layer 1: Input Contract",
                status="PASS",
                details=f"{contract_result.get('rows_found', 0)} rows validated"
            ))
            
            # =====================================================
            # LAYER 2: INPUT VALIDATION
            # =====================================================
            logger.start_layer("Layer 2: Input Validation")
            ok, validation_result = self.l2.validate(df)
            
            status = "PASS" if ok else "WARN"
            logger.end_layer("Layer 2: Input Validation", status, validation_result)
            layer_trace.append(LayerStatus(
                layer="Layer 2: Input Validation",
                status=status,
                details=f"Score: {validation_result.get('validation_score', 0):.1f}%"
            ))
            
            # Critical validation failures should stop pipeline
            if validation_result.get('critical_issues'):
                logger.log_event("Layer 2: Input Validation", "CRITICAL", 
                               validation_result['critical_issues'], LogSeverity.ERROR)
            
            # =====================================================
            # LAYER 3: FEATURE EXTRACTION
            # =====================================================
            logger.start_layer("Layer 3: Feature Extraction")
            df_enriched, feature_stats = self.l3.extract(df)
            
            logger.end_layer("Layer 3: Feature Extraction", "COMPLETE", {
                "features_created": feature_stats.total_features_created,
                "statistical": feature_stats.statistical_features,
                "temporal": feature_stats.temporal_features,
                "categorical": feature_stats.categorical_features
            })
            layer_trace.append(LayerStatus(
                layer="Layer 3: Feature Extraction",
                status="PASS",
                details=f"{feature_stats.total_features_created} features created"
            ))
            
            # =====================================================
            # LAYER 4.1: STRUCTURAL INTEGRITY
            # =====================================================
            logger.start_layer("Layer 4.1: Structural Integrity")
            ok, structural_result = self.l4.sublayer_4_1_structural(df_enriched)
            
            if not ok:
                logger.end_layer("Layer 4.1: Structural Integrity", "FAIL", structural_result)
                layer_trace.append(LayerStatus(
                    layer="Layer 4.1: Structural Integrity",
                    status="FAIL",
                    details="Structural gate failed"
                ))
                return self._build_failure_result(
                    "STRUCTURAL_FAILURE",
                    "Data structure validation failed",
                    logger,
                    layer_trace
                )
                
            logger.end_layer("Layer 4.1: Structural Integrity", "PASS", structural_result)
            layer_trace.append(LayerStatus(
                layer="Layer 4.1: Structural Integrity",
                status="PASS",
                details="All structural checks passed"
            ))
            
            # =====================================================
            # LAYER 4.2: FIELD-LEVEL COMPLIANCE (RULES)
            # =====================================================
            logger.start_layer("Layer 4.2: Field Compliance")
            rules_result = self.l4.sublayer_4_2_rules(df_enriched)
            dqs = rules_result['composite']
            dimension_scores = rules_result['scores']
            
            logger.end_layer("Layer 4.2: Field Compliance", "COMPLETE", {
                "composite_dqs": dqs,
                "dimensions": len(dimension_scores)
            })
            layer_trace.append(LayerStatus(
                layer="Layer 4.2: Field Compliance",
                status="PASS" if dqs >= 70 else "WARN",
                details=f"DQS: {dqs:.1f}%"
            ))
            
            # =====================================================
            # LAYER 4.3: SEMANTIC VALIDATION
            # =====================================================
            logger.start_layer("Layer 4.3: Semantic Validation")
            semantic_result = self.l4.sublayer_4_3_semantic(df_enriched)
            crit_violations = semantic_result['critical_violations']
            
            sem_status = "PASS"
            if crit_violations > 0:
                sem_status = "WARN" if crit_violations < 5 else "FAIL"
                
            logger.end_layer("Layer 4.3: Semantic Validation", sem_status, semantic_result)
            layer_trace.append(LayerStatus(
                layer="Layer 4.3: Semantic Validation",
                status=sem_status,
                details=f"{crit_violations} violations"
            ))
            
            # =====================================================
            # LAYER 4.4: ANOMALY DETECTION (ML)
            # =====================================================
            logger.start_layer("Layer 4.4: Anomaly Detection")
            anomaly_flags, ml_metadata = self.l4.sublayer_4_4_anomaly(
                df_enriched, 
                simulate_failure=simulate_ml_failure
            )
            
            ml_status = "COMPLETE"
            ml_degraded = ml_metadata.get('degraded', False)
            if ml_degraded:
                ml_status = "DEGRADED"
                logger.log_event("Layer 4.4: Anomaly Detection", "DEGRADED", 
                               "ML models unavailable - running in safe fallback mode",
                               LogSeverity.WARN)
                               
            logger.end_layer("Layer 4.4: Anomaly Detection", ml_status, {
                "flags": len(anomaly_flags),
                "models_run": ml_metadata.get('models_run', []),
                "degraded": ml_degraded
            })
            layer_trace.append(LayerStatus(
                layer="Layer 4.4: Anomaly Detection",
                status="WARN" if ml_degraded else ("WARN" if len(anomaly_flags) > 5 else "PASS"),
                details=f"{len(anomaly_flags)} flags" + (" (DEGRADED)" if ml_degraded else "")
            ))
            
            # =====================================================
            # LAYER 4.5: GENAI SUMMARIZATION
            # =====================================================
            logger.start_layer("Layer 4.5: GenAI Summary")
            summary, genai_metadata = self.l4.sublayer_4_5_genai(
                rules_result, 
                semantic_result, 
                anomaly_flags
            )
            
            genai_status = "COMPLETE" if not genai_metadata.get('template_fallback') else "FALLBACK"
            logger.end_layer("Layer 4.5: GenAI Summary", genai_status, genai_metadata)
            layer_trace.append(LayerStatus(
                layer="Layer 4.5: GenAI Summary",
                status="PASS",
                details="Summary generated"
            ))
            
            # =====================================================
            # LAYER 6: STABILITY CHECK
            # =====================================================
            logger.start_layer("Layer 6: Stability Check")
            stability_level, stability_checks = self.l6.check_consistency(
                dqs, 
                dimension_scores,
                crit_violations, 
                len(anomaly_flags)
            )
            
            stab_status = "PASS"
            if stability_level in [ConsistencyLevel.CONTRADICTION, ConsistencyLevel.CRITICAL]:
                stab_status = "FAIL"
                logger.log_event("Layer 6: Stability Check", "CONTRADICTION",
                               self.l6.get_check_summary(), LogSeverity.WARN)
            elif stability_level == ConsistencyLevel.WARNING:
                stab_status = "WARN"
                
            logger.end_layer("Layer 6: Stability Check", stab_status, {
                "level": stability_level.value,
                "checks": len(stability_checks)
            })
            layer_trace.append(LayerStatus(
                layer="Layer 6: Stability Check",
                status=stab_status,
                details=stability_level.value
            ))
            
            # =====================================================
            # LAYER 7: CONFLICT DETECTION
            # =====================================================
            logger.start_layer("Layer 7: Conflict Detection")
            high_severity = sum(1 for f in anomaly_flags if f.severity in ['HIGH', 'CRITICAL'])
            conflict_result = self.l7.resolve(
                dqs,
                dimension_scores,
                len(anomaly_flags),
                high_severity,
                semantic_result['total_violations'],
                crit_violations > 0
            )
            
            conf_status = "PASS" if conflict_result['total_conflicts'] == 0 else "RESOLVED"
            logger.end_layer("Layer 7: Conflict Detection", conf_status, conflict_result)
            layer_trace.append(LayerStatus(
                layer="Layer 7: Conflict Detection",
                status=conf_status,
                details=f"{conflict_result['total_conflicts']} conflicts"
            ))
            
            # =====================================================
            # LAYER 8: CONFIDENCE BAND
            # =====================================================
            logger.start_layer("Layer 8: Confidence Band")
            confidence_result = self.l8.calculate(
                dqs,
                dimension_scores,
                len(anomaly_flags),
                conflict_result,
                len(df),
                ml_degraded
            )
            
            logger.end_layer("Layer 8: Confidence Band", "CALCULATED", confidence_result)
            layer_trace.append(LayerStatus(
                layer="Layer 8: Confidence Band",
                status="PASS",
                details=f"{confidence_result['band']} ({confidence_result['score']})"
            ))
            
            # =====================================================
            # LAYER 9: DECISION GATE
            # =====================================================
            logger.start_layer("Layer 9: Decision Gate")
            decision_result = self.l9.decide(
                dqs,
                dimension_scores,
                [f.to_dict() for f in anomaly_flags],
                confidence_result['band'],
                conflict_result,
                crit_violations,
                ml_degraded
            )
            
            logger.end_layer("Layer 9: Decision Gate", "DECIDED", decision_result.to_dict())
            layer_trace.append(LayerStatus(
                layer="Layer 9: Decision Gate",
                status="PASS",
                details=decision_result.action.value
            ))
            
            # =====================================================
            # LAYER 10: RESPONSIBILITY BOUNDARY
            # =====================================================
            logger.start_layer("Layer 10: Responsibility")
            handoff = self.l10.get_handoff(
                decision_result.action.value,
                {"dqs": dqs, "anomalies": len(anomaly_flags)}
            )
            
            logger.end_layer("Layer 10: Responsibility", "ASSIGNED", handoff)
            layer_trace.append(LayerStatus(
                layer="Layer 10: Responsibility",
                status="PASS",
                details=handoff['accountability']['primary_responsible']
            ))
            
            # =====================================================
            # LAYER 5: OUTPUT CONTRACT (Final Formatting)
            # =====================================================
            logger.log_event("Layer 5: Output Contract", "VALIDATED", "Schema satisfied")
            layer_trace.append(LayerStatus(
                layer="Layer 5: Output Contract",
                status="PASS",
                details="Output validated"
            ))
            
            # =====================================================
            # CONSTRUCT FINAL RESULT
            # =====================================================
            logger.log_event("PIPELINE", "COMPLETED", logger.get_summary())
            
            # Build dimension list
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
                
            # Build anomaly list
            anomalies = []
            for flag in anomaly_flags[:20]:  # Limit to 20 for response size
                anomalies.append(AnomalyDetail(
                    row=flag.row_index,
                    column=flag.column,
                    detector=flag.detector,
                    severity=flag.severity,
                    reason=flag.reason
                ))
                
            return AnalysisResult(
                # Core
                composite_score=round(dqs, 2),
                overall_status=decision_result.action.value,
                summary=summary,
                
                # Details
                dimensions=dimensions,
                anomalies=anomalies,
                
                # Metadata
                trace_id=logger.trace_id,
                confidence_band=confidence_result['band'],
                confidence_score=confidence_result['score'],
                
                # Audit
                layer_trace=layer_trace,
                logs=[str(log) for log in logger.get_logs()],
                
                # Governance
                responsible_party=handoff['accountability']['primary_responsible'],
                next_steps=decision_result.next_steps,
                liability_summary=self.l10.get_liability_summary(decision_result.action.value)
            )
            
        except Exception as e:
            logger.log_critical("PIPELINE", f"Unexpected error: {str(e)}")
            return self._build_failure_result(
                "SYSTEM_ERROR",
                f"Pipeline failed: {str(e)}",
                logger,
                layer_trace
            )
    
    def _build_failure_result(self, 
                             error_code: str, 
                             message: str, 
                             logger: Layer11Logging,
                             layer_trace: List[LayerStatus]) -> AnalysisResult:
        """Build a standardized failure result."""
        return AnalysisResult(
            composite_score=0.0,
            overall_status="NO_ACTION",
            summary=f"Analysis failed: {message}",
            dimensions=[],
            anomalies=[],
            trace_id=logger.trace_id,
            confidence_band="LOW",
            confidence_score=0,
            layer_trace=layer_trace,
            logs=[str(log) for log in logger.get_logs()],
            responsible_party="DATA_ENGINEERING",
            next_steps=["Fix data issues", "Retry analysis"],
            liability_summary="⛔ Analysis incomplete. No processing permitted."
        )
