"""
LAYER 4: MODEL INFERENCE LAYER (AI CONTAINMENT ZONE)
Purpose: Execute ML models within strict safety boundaries.
Type: Hybrid (Rules enforce, ML informs)

Sub-layers:
4.1 - Structural Integrity Gate
4.2 - Field-Level Compliance (7 Dimensions)
4.3 - Semantic Validation
4.4 - Cross-Field Anomaly Detection (ML)
4.5 - GenAI Summarization

Features:
- Safe degradation on ML failure
- Firebreaks between sub-layers
- Detailed scoring for each dimension
- ML model simulation with realistic patterns
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import hashlib
import os

# Google Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

class DimensionStatus(Enum):
    EXCELLENT = "EXCELLENT"  # >= 95
    GOOD = "GOOD"            # >= 85
    ACCEPTABLE = "ACCEPTABLE" # >= 70
    POOR = "POOR"            # >= 50
    CRITICAL = "CRITICAL"    # < 50

@dataclass
class DimensionResult:
    """Result for a single quality dimension."""
    name: str
    score: float
    status: DimensionStatus
    issues_found: int
    details: str
    recommendation: str
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "score": self.score,
            "status": self.status.value,
            "issues_found": self.issues_found,
            "details": self.details,
            "recommendation": self.recommendation
        }

@dataclass
class AnomalyFlag:
    """Single anomaly detection flag."""
    row_index: int
    column: str
    detector: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    reason: str
    value: Any
    
    def to_dict(self) -> Dict:
        return {
            "row": self.row_index,
            "column": self.column,
            "detector": self.detector,
            "severity": self.severity,
            "reason": self.reason,
            "value": str(self.value)
        }

class Layer4Inference:
    """
    LAYER 4: MODEL INFERENCE LAYER (AI CONTAINMENT ZONE)
    
    This layer contains all ML/AI operations with strict boundaries.
    If any ML component fails, the system degrades gracefully to rules-only.
    """
    
    DIMENSION_WEIGHTS = {
        'Completeness': 0.20,
        'Accuracy': 0.15,
        'Validity': 0.15,
        'Uniqueness': 0.15,
        'Consistency': 0.15,
        'Timeliness': 0.10,
        'Integrity': 0.10
    }
    
    def __init__(self):
        self._ml_enabled = True
        self._genai_enabled = True
        self._last_error: Optional[str] = None
        
    # ========================================
    # SUB-LAYER 4.1: STRUCTURAL INTEGRITY GATE
    # ========================================
    def sublayer_4_1_structural(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Binary gate: Pass/Fail.
        Data must meet minimum structural requirements to proceed.
        """
        checks = {
            "non_empty": not df.empty,
            "min_columns": len(df.columns) >= 2,
            "min_rows": len(df) >= 1,
            "readable": True  # If we got here, data is readable
        }
        
        # Critical column check
        critical_columns = ['transaction_id', 'amount']
        for col in critical_columns:
            checks[f"has_{col}"] = col in df.columns
            
        passed = all(checks.values())
        failed_checks = [k for k, v in checks.items() if not v]
        
        return passed, {
            "status": "PASS" if passed else "FAIL",
            "checks": checks,
            "failed_checks": failed_checks,
            "total_columns": len(df.columns),
            "total_rows": len(df)
        }
    
    # ========================================
    # SUB-LAYER 4.2: FIELD-LEVEL COMPLIANCE (RULES)
    # ========================================
    def sublayer_4_2_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Data Quality Score across 7 dimensions.
        Type: 100% Deterministic Rules
        """
        dimensions: List[DimensionResult] = []
        
        # 1. COMPLETENESS - Measure of missing data
        null_counts = df.isnull().sum()
        total_cells = len(df) * len(df.columns)
        null_cells = null_counts.sum()
        completeness_score = round(100 * (1 - null_cells / total_cells), 2)
        
        dimensions.append(DimensionResult(
            name="Completeness",
            score=completeness_score,
            status=self._get_status(completeness_score),
            issues_found=int(null_cells),
            details=f"{null_cells} null values across {len(df.columns)} columns",
            recommendation="Fill missing values or document data gaps" if null_cells > 0 else "Excellent data completeness"
        ))
        
        # 2. ACCURACY - Data correctness (check for obvious errors)
        accuracy_issues = 0
        if 'amount' in df.columns:
            # Negative amounts (should not exist in payments)
            negative_amounts = (pd.to_numeric(df['amount'], errors='coerce') < 0).sum()
            accuracy_issues += negative_amounts
            
        # Check for placeholder values
        placeholder_patterns = ['TEST', 'TEMP', 'XXX', 'N/A', 'TBD']
        for col in df.select_dtypes(include=['object']).columns:
            for pattern in placeholder_patterns:
                accuracy_issues += df[col].astype(str).str.contains(pattern, case=False, na=False).sum()
                
        accuracy_score = round(100 * max(0, 1 - accuracy_issues / len(df)), 2)
        
        dimensions.append(DimensionResult(
            name="Accuracy",
            score=accuracy_score,
            status=self._get_status(accuracy_score),
            issues_found=accuracy_issues,
            details=f"{accuracy_issues} potential data accuracy issues detected",
            recommendation="Review flagged records for data entry errors" if accuracy_issues > 0 else "Data accuracy within acceptable limits"
        ))
        
        # 3. VALIDITY - Conformance to domain rules
        validity_issues = 0
        total_validity_checks = 0
        
        if 'amount' in df.columns:
            total_validity_checks += len(df)
            # Amount must be positive
            invalid_amounts = (pd.to_numeric(df['amount'], errors='coerce') < 0).sum()
            validity_issues += invalid_amounts
            # Amount must be reasonable (< 1M for most transactions)
            extreme_amounts = (pd.to_numeric(df['amount'], errors='coerce') > 1000000).sum()
            validity_issues += extreme_amounts
            
        if 'date' in df.columns:
            total_validity_checks += len(df)
            # Date must be parseable
            parsed_dates = pd.to_datetime(df['date'], errors='coerce')
            invalid_dates = parsed_dates.isna().sum()
            validity_issues += invalid_dates
            # Date must not be in the future
            future_dates = (parsed_dates > pd.Timestamp.now()).sum()
            validity_issues += future_dates
            
        validity_score = round(100 * max(0, 1 - validity_issues / max(total_validity_checks, 1)), 2)
        
        dimensions.append(DimensionResult(
            name="Validity",
            score=validity_score,
            status=self._get_status(validity_score),
            issues_found=validity_issues,
            details=f"{validity_issues} validity violations in {total_validity_checks} checks",
            recommendation="Correct invalid values per business rules" if validity_issues > 0 else "All values within valid ranges"
        ))
        
        # 4. UNIQUENESS - Duplicate detection
        uniqueness_issues = 0
        
        if 'transaction_id' in df.columns:
            duplicate_ids = df['transaction_id'].duplicated().sum()
            uniqueness_issues += duplicate_ids
            
        # Check for fully duplicate rows
        full_duplicates = df.duplicated().sum()
        uniqueness_issues += full_duplicates
        
        uniqueness_score = round(100 * max(0, 1 - uniqueness_issues / len(df)), 2)
        
        dimensions.append(DimensionResult(
            name="Uniqueness",
            score=uniqueness_score,
            status=self._get_status(uniqueness_score),
            issues_found=uniqueness_issues,
            details=f"{uniqueness_issues} duplicate records found",
            recommendation="Remove or merge duplicate records" if uniqueness_issues > 0 else "No duplicates detected"
        ))
        
        # 5. CONSISTENCY - Cross-field consistency
        consistency_issues = 0
        
        # Check for logical consistency (e.g., dates in order, amounts match totals)
        if 'date' in df.columns and len(df) > 1:
            dates = pd.to_datetime(df['date'], errors='coerce').sort_values()
            # Check for chronological order violations if there's a sequence
            
        if 'status' in df.columns and 'amount' in df.columns:
            # Failed transactions should have specific amount patterns
            failed_with_high_amount = (
                (df['status'] == 'FAILED') & 
                (pd.to_numeric(df['amount'], errors='coerce') > 10000)
            ).sum()
            consistency_issues += failed_with_high_amount
            
        consistency_score = round(100 * max(0, 1 - consistency_issues / max(len(df), 1)), 2)
        
        dimensions.append(DimensionResult(
            name="Consistency",
            score=consistency_score,
            status=self._get_status(consistency_score),
            issues_found=consistency_issues,
            details=f"{consistency_issues} cross-field consistency issues",
            recommendation="Review records with logical inconsistencies" if consistency_issues > 0 else "Data is internally consistent"
        ))
        
        # 6. TIMELINESS - Data freshness
        timeliness_score = 100.0
        timeliness_issues = 0
        
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'], errors='coerce').dropna()
            if len(dates) > 0:
                latest_date = dates.max()
                days_old = (pd.Timestamp.now() - latest_date).days
                
                # Penalize based on age
                if days_old > 365:
                    timeliness_score = 50.0
                    timeliness_issues = days_old
                elif days_old > 90:
                    timeliness_score = 70.0
                    timeliness_issues = days_old
                elif days_old > 30:
                    timeliness_score = 85.0
                    timeliness_issues = days_old
                elif days_old > 7:
                    timeliness_score = 95.0
                    
        dimensions.append(DimensionResult(
            name="Timeliness",
            score=timeliness_score,
            status=self._get_status(timeliness_score),
            issues_found=timeliness_issues,
            details=f"Data is {timeliness_issues} days old" if timeliness_issues > 0 else "Data is current",
            recommendation="Update data source" if timeliness_issues > 30 else "Data freshness acceptable"
        ))
        
        # 7. INTEGRITY - Referential and structural integrity
        integrity_score = 100.0
        integrity_issues = 0
        
        # Check for orphaned references (if we had foreign keys)
        # Check for structural anomalies
        if 'transaction_id' in df.columns:
            # ID format check (should be consistent)
            id_lengths = df['transaction_id'].astype(str).str.len()
            if id_lengths.std() > 5:  # High variance in ID lengths
                integrity_issues += 1
                integrity_score -= 10
                
        dimensions.append(DimensionResult(
            name="Integrity",
            score=max(0, integrity_score),
            status=self._get_status(integrity_score),
            issues_found=integrity_issues,
            details=f"{integrity_issues} structural integrity concerns",
            recommendation="Standardize data formats" if integrity_issues > 0 else "Data structure is sound"
        ))
        
        # Calculate weighted composite score
        composite_score = sum(
            dim.score * self.DIMENSION_WEIGHTS.get(dim.name, 0.1)
            for dim in dimensions
        )
        
        return {
            "scores": {dim.name: dim.score for dim in dimensions},
            "dimensions": [dim.to_dict() for dim in dimensions],
            "composite": round(composite_score, 2),
            "weights": self.DIMENSION_WEIGHTS
        }
    
    # ========================================
    # SUB-LAYER 4.3: SEMANTIC VALIDATION
    # ========================================
    def sublayer_4_3_semantic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Domain-specific semantic validation.
        Checks business logic beyond pure data quality.
        """
        violations = []
        critical_count = 0
        
        # Rule 1: High-value transactions in low-value categories
        if 'amount' in df.columns and 'merchant_category' in df.columns:
            low_value_categories = ['Coffee', 'Snacks', 'Parking', 'Transit']
            for idx, row in df.iterrows():
                if row.get('merchant_category') in low_value_categories:
                    amount = pd.to_numeric(row.get('amount', 0), errors='coerce')
                    if amount > 100:
                        violations.append({
                            "rule": "HIGH_VALUE_LOW_CATEGORY",
                            "row": idx,
                            "severity": "MEDIUM",
                            "details": f"${amount} in {row.get('merchant_category')}"
                        })
                        
        # Rule 2: Suspiciously round amounts (exactly $1000, $5000, etc.)
        if 'amount' in df.columns:
            amounts = pd.to_numeric(df['amount'], errors='coerce')
            round_thousands = amounts[amounts % 1000 == 0]
            if len(round_thousands) > len(df) * 0.3:  # More than 30% round
                violations.append({
                    "rule": "SUSPICIOUS_ROUND_AMOUNTS",
                    "severity": "HIGH",
                    "details": f"{len(round_thousands)} suspiciously round amounts"
                })
                critical_count += 1
                
        # Rule 3: Amount vs Category rationality (extreme outliers)
        if 'amount' in df.columns:
            amounts = pd.to_numeric(df['amount'], errors='coerce')
            extreme_high = (amounts > 50000).sum()
            if extreme_high > 0:
                critical_count += extreme_high
                violations.append({
                    "rule": "EXTREME_AMOUNT",
                    "severity": "CRITICAL",
                    "details": f"{extreme_high} transactions exceed $50,000"
                })
                
        # Rule 4: Weekend/Holiday fraud patterns
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'], errors='coerce')
            weekend_txns = dates.dt.dayofweek.isin([5, 6]).sum()
            total_txns = len(dates.dropna())
            if total_txns > 0 and weekend_txns / total_txns > 0.5:
                violations.append({
                    "rule": "HIGH_WEEKEND_RATIO",
                    "severity": "LOW",
                    "details": f"{weekend_txns}/{total_txns} transactions on weekends"
                })
        
        return {
            "score": max(0, 100 - len(violations) * 5),
            "critical_violations": critical_count,
            "total_violations": len(violations),
            "violations": violations
        }
    
    # ========================================
    # SUB-LAYER 4.4: CROSS-FIELD ANOMALY DETECTION (ML)
    # ========================================
    def sublayer_4_4_anomaly(self, df: pd.DataFrame, 
                             simulate_failure: bool = False) -> Tuple[List[AnomalyFlag], Dict]:
        """
        ML-based anomaly detection.
        Has FIREBREAK: Returns empty on failure (safe degradation).
        """
        flags: List[AnomalyFlag] = []
        ml_metadata = {
            "ml_enabled": self._ml_enabled,
            "models_run": [],
            "degraded": False
        }
        
        # FIREBREAK: Simulate ML failure
        if simulate_failure:
            self._last_error = "ML_MODEL_UNAVAILABLE"
            ml_metadata["degraded"] = True
            ml_metadata["error"] = "Model file corrupted - safe degradation active"
            return [], ml_metadata
            
        try:
            # ----- DETECTOR A: Statistical Outlier (Isolation Forest Simulation) -----
            if 'amount' in df.columns:
                amounts = pd.to_numeric(df['amount'], errors='coerce').dropna()
                if len(amounts) > 0:
                    mean = amounts.mean()
                    std = amounts.std() if len(amounts) > 1 else 1
                    
                    # Z-score based detection
                    z_scores = (amounts - mean) / max(std, 0.001)
                    outlier_threshold = 2.5  # 2.5 sigma
                    
                    for idx in amounts[abs(z_scores) > outlier_threshold].index:
                        flags.append(AnomalyFlag(
                            row_index=int(idx),
                            column="amount",
                            detector="IsolationForest",
                            severity="HIGH" if abs(z_scores[idx]) > 3 else "MEDIUM",
                            reason=f"Amount {z_scores[idx]:.1f} sigma from mean",
                            value=df.loc[idx, 'amount']
                        ))
                        
                ml_metadata["models_run"].append("IsolationForest")
                
            # ----- DETECTOR B: Association Rule Mining Simulation -----
            if 'merchant_category' in df.columns and 'amount' in df.columns:
                # Find unusual category-amount combinations
                category_stats = df.groupby('merchant_category')['amount'].agg(['mean', 'std']).reset_index()
                
                for idx, row in df.iterrows():
                    cat = row.get('merchant_category')
                    amt = pd.to_numeric(row.get('amount', 0), errors='coerce')
                    
                    cat_stat = category_stats[category_stats['merchant_category'] == cat]
                    if len(cat_stat) > 0:
                        cat_mean = cat_stat['mean'].values[0]
                        cat_std = cat_stat['std'].values[0] or 1
                        
                        if abs(amt - cat_mean) > 2 * cat_std:
                            flags.append(AnomalyFlag(
                                row_index=int(idx),
                                column="amount+merchant_category",
                                detector="AssociationRules",
                                severity="MEDIUM",
                                reason=f"Unusual amount for category {cat}",
                                value=f"${amt} in {cat}"
                            ))
                            
                ml_metadata["models_run"].append("AssociationRules")
                
            # ----- DETECTOR C: Temporal Pattern Analysis -----
            if 'date' in df.columns:
                dates = pd.to_datetime(df['date'], errors='coerce')
                valid_dates = dates.dropna()
                
                if len(valid_dates) > 0:
                    # Check for burst patterns (many transactions in short time)
                    date_counts = valid_dates.dt.date.value_counts()
                    burst_threshold = date_counts.mean() + 2 * date_counts.std()
                    
                    burst_days = date_counts[date_counts > burst_threshold]
                    for day in burst_days.index:
                        day_rows = df[dates.dt.date == day].index
                        for idx in day_rows[:3]:  # Limit flagging
                            flags.append(AnomalyFlag(
                                row_index=int(idx),
                                column="date",
                                detector="TemporalAnalysis",
                                severity="LOW",
                                reason=f"Burst activity detected on {day}",
                                value=str(day)
                            ))
                            
                ml_metadata["models_run"].append("TemporalAnalysis")
                
        except Exception as e:
            # FIREBREAK: Safe degradation
            self._last_error = str(e)
            ml_metadata["degraded"] = True
            ml_metadata["error"] = f"ML detection failed: {str(e)}"
            return [], ml_metadata
            
        ml_metadata["total_flags"] = len(flags)
        ml_metadata["high_severity"] = sum(1 for f in flags if f.severity == "HIGH")
        
        return flags, ml_metadata
    
    # ========================================
    # SUB-LAYER 4.5: GENAI SUMMARIZATION
    # ========================================
    def sublayer_4_5_genai(self, rules_result: Dict, semantic_result: Dict, 
                          anomaly_flags: List[AnomalyFlag],
                          simulate_failure: bool = False) -> Tuple[str, Dict]:
        """
        Generate executive summary using GenAI (Google Gemini).
        Has FIREBREAK: Falls back to template on failure.
        """
        metadata = {
            "genai_enabled": self._genai_enabled,
            "template_fallback": False,
            "gemini_available": GEMINI_AVAILABLE
        }
        
        dqs = rules_result.get('composite', 0)
        violations = semantic_result.get('critical_violations', 0)
        anomaly_count = len(anomaly_flags)
        dimensions = rules_result.get('dimensions', [])
        
        # FIREBREAK: Simulate GenAI failure
        if simulate_failure:
            metadata["template_fallback"] = True
            metadata["reason"] = "Simulation mode"
            summary = self._generate_template_summary(dqs, violations, anomaly_count)
            return summary, metadata
        
        # Try to use Google Gemini API
        api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        
        if GEMINI_AVAILABLE and api_key:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                # Build detailed prompt
                prompt = self._build_gemini_prompt(dqs, dimensions, anomaly_flags, violations)
                
                # Call Gemini API
                response = model.generate_content(prompt)
                summary = response.text
                
                metadata["model"] = "gemini-2.0-flash"
                metadata["api_used"] = True
                
                return summary, metadata
                
            except Exception as e:
                # FIREBREAK: Fall back to template on API error
                metadata["template_fallback"] = True
                metadata["error"] = str(e)
                metadata["reason"] = "API call failed"
        else:
            metadata["template_fallback"] = True
            if not GEMINI_AVAILABLE:
                metadata["reason"] = "google-generativeai not installed"
            elif not api_key:
                metadata["reason"] = "GEMINI_API_KEY not set"
        
        # Fallback to enhanced template
        summary = self._generate_enhanced_template_summary(dqs, dimensions, anomaly_flags, violations)
        metadata["model"] = "template_v3_enhanced"
        
        return summary, metadata
    
    def _build_gemini_prompt(self, dqs: float, dimensions: List, 
                             anomaly_flags: List, violations: int) -> str:
        """Build a detailed prompt for Gemini API."""
        # Format dimension scores
        dim_text = "\n".join([
            f"  - {d.get('name', 'Unknown')}: {d.get('score', 0):.1f}% ({d.get('status', 'N/A')})"
            for d in dimensions
        ])
        
        # Format anomaly summary
        high_sev = sum(1 for f in anomaly_flags if f.severity in ['HIGH', 'CRITICAL'])
        med_sev = sum(1 for f in anomaly_flags if f.severity == 'MEDIUM')
        low_sev = sum(1 for f in anomaly_flags if f.severity == 'LOW')
        
        prompt = f"""You are a Data Quality Expert AI. Analyze this payment dataset quality report and provide an executive summary.

## DATA QUALITY REPORT

**Composite DQS Score**: {dqs:.1f}/100

**Dimension Breakdown**:
{dim_text}

**Anomaly Detection Results**:
- Total Anomalies Flagged: {len(anomaly_flags)}
- High Severity: {high_sev}
- Medium Severity: {med_sev}
- Low Severity: {low_sev}

**Semantic Violations**: {violations} critical business rule violations

## INSTRUCTIONS
Provide a concise 3-4 sentence executive summary that:
1. States the overall data quality assessment (Excellent/Acceptable/Concerning/Critical)
2. Highlights the most important finding (best or worst dimension)
3. Notes any anomalies or violations that need attention
4. Gives a clear recommendation (Safe to Use / Review Required / Escalate / Do Not Use)

Keep the response professional and actionable. Do not use markdown formatting."""
        
        return prompt
    
    def _generate_enhanced_template_summary(self, dqs: float, dimensions: List,
                                           anomaly_flags: List, violations: int) -> str:
        """Generate enhanced template summary when Gemini is unavailable."""
        parts = []
        
        # Overall assessment
        if dqs >= 90:
            parts.append(f"Excellent Data Quality (DQS: {dqs:.1f}/100).")
            assessment = "production-ready"
        elif dqs >= 75:
            parts.append(f"Good Data Quality (DQS: {dqs:.1f}/100).")
            assessment = "acceptable with minor issues"
        elif dqs >= 60:
            parts.append(f"Acceptable Data Quality (DQS: {dqs:.1f}/100).")
            assessment = "requires review"
        elif dqs >= 40:
            parts.append(f"Poor Data Quality (DQS: {dqs:.1f}/100).")
            assessment = "significant issues detected"
        else:
            parts.append(f"Critical Data Quality Issues (DQS: {dqs:.1f}/100).")
            assessment = "not suitable for use"
        
        # Find worst dimension
        if dimensions:
            worst = min(dimensions, key=lambda d: d.get('score', 100))
            if worst.get('score', 100) < 80:
                parts.append(f"Lowest dimension: {worst.get('name', 'Unknown')} at {worst.get('score', 0):.1f}%.")
        
        # Anomaly summary
        if len(anomaly_flags) > 0:
            high_sev = sum(1 for f in anomaly_flags if f.severity in ['HIGH', 'CRITICAL'])
            if high_sev > 0:
                parts.append(f"{high_sev} high-severity anomalies require investigation.")
            else:
                parts.append(f"{len(anomaly_flags)} minor anomalies flagged.")
        
        # Violations
        if violations > 0:
            parts.append(f"{violations} business rule violations detected.")
        
        # Recommendation
        if dqs >= 90 and violations == 0 and len(anomaly_flags) < 3:
            parts.append("Recommendation: Safe to use for automated processing.")
        elif dqs >= 70 and violations == 0:
            parts.append("Recommendation: Review flagged items before use.")
        elif violations > 0:
            parts.append("Recommendation: Escalate to data governance team.")
        else:
            parts.append("Recommendation: Do not use until issues are resolved.")
        
        return " ".join(parts)
    
    def _generate_template_summary(self, dqs: float, violations: int, anomalies: int) -> str:
        """Fallback template summary."""
        return f"Dataset Quality Score: {dqs:.1f}/100. {violations} semantic violations. {anomalies} ML anomalies. {'Review required.' if dqs < 80 or violations > 0 else 'Safe for use.'}"
    
    def _get_recommendation(self, dqs: float, violations: int, anomalies: int) -> str:
        """Generate actionable recommendation."""
        if dqs >= 90 and violations == 0 and anomalies < 3:
            return "Data is production-ready. Proceed with automated processing."
        elif dqs >= 70 and violations == 0:
            return "Review flagged anomalies. Consider remediation for quality dimensions below 80%."
        elif violations > 0:
            return "Do not use until semantic violations are resolved. Escalate to data governance team."
        else:
            return "Significant remediation required. Consider data refresh or source investigation."
    
    def _get_status(self, score: float) -> DimensionStatus:
        """Convert score to status enum."""
        if score >= 95:
            return DimensionStatus.EXCELLENT
        elif score >= 85:
            return DimensionStatus.GOOD
        elif score >= 70:
            return DimensionStatus.ACCEPTABLE
        elif score >= 50:
            return DimensionStatus.POOR
        else:
            return DimensionStatus.CRITICAL
