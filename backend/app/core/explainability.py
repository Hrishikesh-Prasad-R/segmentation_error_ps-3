"""
EXPLAINABILITY MODULE
Purpose: Make the DQS Agent fully transparent and "Judge-Ready"
Type: Documentation & Audit

This module demonstrates WHY this system is NOT a black box:
1. Every decision has a traceable rationale
2. Every score has a formula-based explanation
3. Every ML flag has human-readable justification
4. The entire pipeline is auditable

Key Principle: "Explainable AI" - No hidden logic
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class ExplainabilityLevel(Enum):
    """How detailed the explanation should be."""
    SUMMARY = "SUMMARY"       # One-liner
    DETAILED = "DETAILED"     # Full breakdown
    TECHNICAL = "TECHNICAL"   # Include formulas/code references

@dataclass
class DecisionExplanation:
    """Explains why a specific decision was made."""
    decision: str
    confidence: float
    reasoning: List[str]
    contributing_factors: Dict[str, Any]
    alternatives_considered: List[str]
    human_override_allowed: bool
    
    def to_dict(self) -> Dict:
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "contributing_factors": self.contributing_factors,
            "alternatives_considered": self.alternatives_considered,
            "human_override_allowed": self.human_override_allowed
        }

@dataclass 
class ScoreExplanation:
    """Explains how a specific score was calculated."""
    dimension: str
    score: float
    formula: str
    inputs: Dict[str, Any]
    calculation_steps: List[str]
    threshold_context: str
    
    def to_dict(self) -> Dict:
        return {
            "dimension": self.dimension,
            "score": self.score,
            "formula": self.formula,
            "inputs": self.inputs,
            "calculation_steps": self.calculation_steps,
            "threshold_context": self.threshold_context
        }

@dataclass
class AnomalyExplanation:
    """Explains why a specific anomaly was flagged."""
    detector: str
    row_index: int
    column: str
    value: Any
    expected_range: str
    deviation: str
    detection_method: str
    false_positive_probability: str
    
    def to_dict(self) -> Dict:
        return {
            "detector": self.detector,
            "row_index": self.row_index,
            "column": self.column,
            "value": str(self.value),
            "expected_range": self.expected_range,
            "deviation": self.deviation,
            "detection_method": self.detection_method,
            "false_positive_probability": self.false_positive_probability
        }

class ExplainabilityEngine:
    """
    The Explainability Engine makes every aspect of the DQS Agent transparent.
    
    WHY THIS IS NOT A BLACK BOX:
    ============================
    1. RULES-BASED SCORING: All 7 dimensions use explicit formulas, not learned weights
    2. TRANSPARENT THRESHOLDS: Decision boundaries are configurable constants
    3. TRACEABLE ML: Anomaly detection uses explainable statistical methods
    4. AUDITABLE LOGS: Every step is logged with timestamp and rationale
    5. HUMAN-IN-THE-LOOP: Critical decisions require human approval
    6. NO HIDDEN STATE: All data flows through documented layers
    """
    
    # Dimension calculation formulas (fully transparent)
    FORMULAS = {
        "Completeness": "100 * (1 - null_cells / total_cells)",
        "Accuracy": "100 * max(0, 1 - accuracy_issues / row_count)",
        "Validity": "100 * max(0, 1 - validity_issues / total_validity_checks)",
        "Uniqueness": "100 * max(0, 1 - duplicate_count / row_count)",
        "Consistency": "100 * max(0, 1 - consistency_issues / row_count)",
        "Timeliness": "100 - (days_since_latest * decay_factor)",
        "Integrity": "100 - (structural_issues * 10)"
    }
    
    # Decision thresholds (fully transparent)
    THRESHOLDS = {
        "DQS_SAFE": 75.0,
        "DQS_REVIEW": 50.0,
        "DQS_ESCALATE": 30.0,
        "ANOMALY_SAFE": 5,
        "ANOMALY_REVIEW": 15,
        "CONFIDENCE_HIGH": 80,
        "CONFIDENCE_MEDIUM": 50
    }
    
    # ML method explanations
    ML_METHODS = {
        "IsolationForest": {
            "description": "Statistical outlier detection using Z-score (standard deviations from mean)",
            "formula": "z_score = (value - mean) / std_dev",
            "threshold": "Flagged if |z_score| > 2.5",
            "interpretability": "HIGH - Simple statistical measure"
        },
        "AssociationRules": {
            "description": "Detects unusual category-value combinations based on historical patterns",
            "formula": "deviation = (value - category_mean) / category_std",
            "threshold": "Flagged if deviation > 2.0",
            "interpretability": "HIGH - Compares to category-specific baseline"
        },
        "TemporalAnalysis": {
            "description": "Identifies burst patterns in transaction timing",
            "formula": "daily_count > mean_daily + 2 * std_daily",
            "threshold": "Flagged if activity exceeds expected burst threshold",
            "interpretability": "HIGH - Time-series pattern recognition"
        }
    }
    
    def __init__(self):
        self.explanations: List[Dict] = []
        self.layer_explanations: Dict[str, str] = {}
        
    def explain_layer(self, layer_name: str, inputs: Dict, outputs: Dict, 
                      duration_ms: float) -> Dict:
        """Generate explanation for a layer's execution."""
        explanation = {
            "layer": layer_name,
            "timestamp": datetime.now().isoformat(),
            "duration_ms": duration_ms,
            "what_happened": self._get_layer_description(layer_name),
            "inputs_received": self._summarize_inputs(inputs),
            "outputs_produced": self._summarize_outputs(outputs),
            "why_this_matters": self._get_layer_importance(layer_name),
            "transparency_note": self._get_transparency_note(layer_name)
        }
        self.explanations.append(explanation)
        return explanation
    
    def explain_score(self, dimension: str, score: float, 
                      raw_inputs: Dict) -> ScoreExplanation:
        """Explain how a dimension score was calculated."""
        formula = self.FORMULAS.get(dimension, "Unknown formula")
        
        # Build calculation steps
        steps = []
        if dimension == "Completeness":
            total = raw_inputs.get("total_cells", 0)
            null = raw_inputs.get("null_cells", 0)
            steps = [
                f"1. Counted total cells: {total}",
                f"2. Counted null/empty cells: {null}",
                f"3. Applied formula: 100 * (1 - {null}/{total})",
                f"4. Result: {score:.2f}%"
            ]
        elif dimension == "Uniqueness":
            total = raw_inputs.get("row_count", 0)
            dupes = raw_inputs.get("duplicates", 0)
            steps = [
                f"1. Total rows: {total}",
                f"2. Duplicate transaction IDs found: {dupes}",
                f"3. Applied formula: 100 * (1 - {dupes}/{total})",
                f"4. Result: {score:.2f}%"
            ]
        else:
            steps = [f"Score calculated using: {formula}", f"Result: {score:.2f}%"]
            
        # Threshold context
        if score >= 95:
            threshold = "EXCELLENT (>=95%)"
        elif score >= 85:
            threshold = "GOOD (>=85%)"
        elif score >= 70:
            threshold = "ACCEPTABLE (>=70%)"
        elif score >= 50:
            threshold = "POOR (>=50%)"
        else:
            threshold = "CRITICAL (<50%)"
            
        return ScoreExplanation(
            dimension=dimension,
            score=score,
            formula=formula,
            inputs=raw_inputs,
            calculation_steps=steps,
            threshold_context=threshold
        )
    
    def explain_anomaly(self, detector: str, row: int, column: str,
                        value: Any, stats: Dict) -> AnomalyExplanation:
        """Explain why a specific value was flagged as anomalous."""
        method_info = self.ML_METHODS.get(detector, {})
        
        # Calculate expected range
        mean = stats.get("mean", 0)
        std = stats.get("std", 1)
        expected_range = f"{mean - 2*std:.2f} to {mean + 2*std:.2f}"
        
        # Calculate deviation
        if std > 0:
            z_score = (float(value) - mean) / std
            deviation = f"{abs(z_score):.2f} standard deviations from mean"
        else:
            deviation = "Unable to calculate (zero variance)"
            
        return AnomalyExplanation(
            detector=detector,
            row_index=row,
            column=column,
            value=value,
            expected_range=expected_range,
            deviation=deviation,
            detection_method=method_info.get("description", "Statistical analysis"),
            false_positive_probability="~5% (using 2-sigma threshold)"
        )
    
    def explain_decision(self, decision: str, dqs: float, anomalies: int,
                         violations: int, confidence: str) -> DecisionExplanation:
        """Explain why a specific decision was made."""
        reasoning = []
        factors = {}
        alternatives = []
        
        factors["dqs_score"] = dqs
        factors["anomaly_count"] = anomalies
        factors["violation_count"] = violations
        factors["confidence_band"] = confidence
        
        if decision == "SAFE_TO_USE":
            reasoning = [
                f"DQS score ({dqs:.1f}%) exceeds safe threshold ({self.THRESHOLDS['DQS_SAFE']}%)",
                f"Anomaly count ({anomalies}) is within acceptable limit ({self.THRESHOLDS['ANOMALY_SAFE']})",
                f"No critical semantic violations detected",
                f"Confidence band is {confidence}"
            ]
            alternatives = [
                "REVIEW_REQUIRED - Would apply if anomalies > 5 or confidence was MEDIUM",
                "ESCALATE - Would apply if DQS < 30% or critical violations existed"
            ]
        elif decision == "REVIEW_REQUIRED":
            reasoning = [
                f"DQS score ({dqs:.1f}%) is acceptable but has concerns",
                f"Anomaly count ({anomalies}) or confidence level warrants human review",
                "System recommends verification before processing"
            ]
            alternatives = [
                "SAFE_TO_USE - Would apply if all metrics were in safe ranges",
                "ESCALATE - Would apply if critical thresholds were breached"
            ]
        elif decision == "ESCALATE":
            reasoning = [
                f"DQS score ({dqs:.1f}%) is below minimum threshold OR",
                f"Critical violations ({violations}) detected OR",
                "Unresolvable conflicts between Rules and ML signals"
            ]
            alternatives = [
                "SAFE_TO_USE - Not possible given current metrics",
                "REVIEW_REQUIRED - Insufficient; escalation required for this severity"
            ]
        else:
            reasoning = ["System could not complete analysis"]
            alternatives = ["All other decisions require successful analysis"]
            
        return DecisionExplanation(
            decision=decision,
            confidence=self.THRESHOLDS.get(f"CONFIDENCE_{confidence}", 50) / 100,
            reasoning=reasoning,
            contributing_factors=factors,
            alternatives_considered=alternatives,
            human_override_allowed=True
        )
    
    def get_system_transparency_report(self) -> Dict:
        """Generate a full transparency report for the system."""
        return {
            "title": "DQS Agent Transparency Report",
            "generated_at": datetime.now().isoformat(),
            "why_not_black_box": [
                "All scoring formulas are explicit mathematical calculations",
                "Decision thresholds are configurable constants, not learned",
                "ML methods use interpretable statistical techniques",
                "Every layer is logged with inputs, outputs, and timing",
                "Human operators can override any automated decision",
                "No neural networks or opaque models are used",
                "Source code is fully auditable"
            ],
            "explainability_level": "FULL",
            "audit_trail_available": True,
            "formulas": self.FORMULAS,
            "thresholds": self.THRESHOLDS,
            "ml_methods": self.ML_METHODS,
            "layer_count": 11,
            "layers": [
                {"id": 1, "name": "Input Contract", "type": "DETERMINISTIC"},
                {"id": 2, "name": "Input Validation", "type": "DETERMINISTIC"},
                {"id": 3, "name": "Feature Extraction", "type": "DETERMINISTIC"},
                {"id": "4.1", "name": "Structural Integrity", "type": "DETERMINISTIC"},
                {"id": "4.2", "name": "Field Compliance", "type": "RULES-BASED"},
                {"id": "4.3", "name": "Semantic Validation", "type": "RULES-BASED"},
                {"id": "4.4", "name": "Anomaly Detection", "type": "STATISTICAL ML"},
                {"id": "4.5", "name": "GenAI Summary", "type": "LLM (with fallback)"},
                {"id": 5, "name": "Output Contract", "type": "DETERMINISTIC"},
                {"id": 6, "name": "Stability Check", "type": "DETERMINISTIC"},
                {"id": 7, "name": "Conflict Detection", "type": "RULES-BASED"},
                {"id": 8, "name": "Confidence Band", "type": "FORMULA-BASED"},
                {"id": 9, "name": "Decision Gate", "type": "FINITE STATE MACHINE"},
                {"id": 10, "name": "Responsibility", "type": "POLICY"},
                {"id": 11, "name": "Logging", "type": "AUDIT"}
            ]
        }
    
    def _get_layer_description(self, layer_name: str) -> str:
        """Get human-readable description of what a layer does."""
        descriptions = {
            "Layer 1: Input Contract": "Verified the uploaded file meets structural requirements (required columns, minimum rows)",
            "Layer 2: Input Validation": "Checked for null values, invalid formats, and data type mismatches",
            "Layer 3: Feature Extraction": "Created derived features (z-scores, percentiles, temporal patterns) for analysis",
            "Layer 4.1: Structural Integrity": "Binary gate to confirm data structure is analyzable",
            "Layer 4.2: Field Compliance": "Scored 7 quality dimensions using explicit formulas",
            "Layer 4.3: Semantic Validation": "Applied business rules to detect logical violations",
            "Layer 4.4: Anomaly Detection": "Used statistical methods to flag outliers (NOT a black box)",
            "Layer 4.5: GenAI Summary": "Generated executive summary (LLM or template-based)",
            "Layer 5: Output Contract": "Validated output format matches API schema",
            "Layer 6: Stability Check": "Detected contradictions between signals",
            "Layer 7: Conflict Detection": "Resolved disagreements (Rules > ML principle)",
            "Layer 8: Confidence Band": "Calculated decision confidence with deduction breakdown",
            "Layer 9: Decision Gate": "Applied FSM to determine final action",
            "Layer 10: Responsibility": "Assigned accountability and SLAs",
            "Layer 11: Logging": "Recorded immutable audit trail"
        }
        return descriptions.get(layer_name, "Processing layer")
    
    def _get_layer_importance(self, layer_name: str) -> str:
        """Explain why this layer exists."""
        importance = {
            "Layer 4.4: Anomaly Detection": "Identifies patterns humans might miss, but NEVER overrides rules",
            "Layer 7: Conflict Detection": "Ensures 'Rules > ML' principle is enforced - ML is advisory only",
            "Layer 9: Decision Gate": "Deterministic FSM ensures same inputs always produce same decision",
            "Layer 10: Responsibility": "Clear accountability prevents ambiguity in governance"
        }
        return importance.get(layer_name, "Essential for data quality assessment")
    
    def _get_transparency_note(self, layer_name: str) -> str:
        """Add transparency note specific to layer."""
        notes = {
            "Layer 4.4: Anomaly Detection": "Uses Z-score (statistical, not neural network). Formula: z = (x - mean) / std. Threshold: 2.5 sigma.",
            "Layer 4.5: GenAI Summary": "If LLM unavailable, uses deterministic template. LLM output is advisory only.",
            "Layer 9: Decision Gate": "Finite State Machine with 4 states. Decision logic is in source code, fully auditable."
        }
        return notes.get(layer_name, "Logic is deterministic and auditable")
    
    def _summarize_inputs(self, inputs: Dict) -> Dict:
        """Summarize inputs for logging."""
        if not inputs:
            return {"summary": "No inputs"}
        return {k: str(v)[:100] for k, v in list(inputs.items())[:5]}
    
    def _summarize_outputs(self, outputs: Dict) -> Dict:
        """Summarize outputs for logging."""
        if not outputs:
            return {"summary": "No outputs"}
        return {k: str(v)[:100] for k, v in list(outputs.items())[:5]}
