"""
COMPREHENSIVE TEST SUITE
Purpose: Validate all 11 layers work correctly and are properly integrated.

Test Categories:
1. Unit Tests - Individual layer functionality
2. Integration Tests - Layer interactions
3. Edge Case Tests - Boundary conditions
4. Failure Mode Tests - Safe degradation
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.agent import PipelineOrchestrator, AnalysisResult
from app.core.layers.layer_1_2_input import Layer1InputContract, Layer2InputValidation
from app.core.layers.layer_3_features import Layer3FeatureExtraction
from app.core.layers.layer_4_inference import Layer4Inference
from app.core.layers.layer_5_6_output import Layer5OutputContract, Layer6Stability, ConsistencyLevel
from app.core.layers.layer_7_8_conflict import Layer7ConflictDetection, Layer8ConfidenceBand
from app.core.layers.layer_9_decision import Layer9DecisionGate, DecisionAction
from app.core.layers.layer_10_responsibility import Layer10Responsibility
from app.core.layers.layer_11_logging import Layer11Logging, LogSeverity


class TestResult:
    """Simple test result container."""
    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message
        
    def __str__(self):
        status = "[PASS]" if self.passed else "[FAIL]"
        return f"{status}: {self.name}" + (f" - {self.message}" if self.message else "")


class TestSuite:
    """Test suite for the 11-Layer DQS Agent."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.pipeline = PipelineOrchestrator()
        
    def run_all_tests(self) -> List[TestResult]:
        """Run all tests and return results."""
        print("\n" + "="*60)
        print("[TEST] RUNNING DQS AGENT TEST SUITE")
        print("="*60 + "\n")
        
        # Layer 1 & 2 Tests
        self._test_layer_1_empty_data()
        self._test_layer_1_missing_columns()
        self._test_layer_1_valid_contract()
        self._test_layer_2_null_validation()
        self._test_layer_2_duplicate_detection()
        
        # Layer 3 Tests
        self._test_layer_3_feature_extraction()
        self._test_layer_3_edge_cases()
        
        # Layer 4 Tests
        self._test_layer_4_structural_gate()
        self._test_layer_4_rules_scoring()
        self._test_layer_4_semantic_validation()
        self._test_layer_4_anomaly_detection()
        self._test_layer_4_ml_degradation()
        
        # Layer 5 & 6 Tests
        self._test_layer_6_consistency_checks()
        
        # Layer 7 & 8 Tests
        self._test_layer_7_conflict_detection()
        self._test_layer_8_confidence_calculation()
        
        # Layer 9 Tests
        self._test_layer_9_decision_safe()
        self._test_layer_9_decision_escalate()
        
        # Layer 10 Tests
        self._test_layer_10_responsibility()
        
        # Layer 11 Tests
        self._test_layer_11_logging()
        
        # Integration Tests
        self._test_full_pipeline_clean_data()
        self._test_full_pipeline_problematic_data()
        self._test_full_pipeline_ml_failure()
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    # =========================================================
    # LAYER 1 & 2 TESTS
    # =========================================================
    
    def _test_layer_1_empty_data(self):
        """Test Layer 1 rejects empty data."""
        l1 = Layer1InputContract()
        df = pd.DataFrame()
        ok, result = l1.check(df)
        
        self.results.append(TestResult(
            "Layer 1: Rejects empty DataFrame",
            ok == False and result['code'] == 'EMPTY_FILE',
            result.get('message', '')
        ))
        
    def _test_layer_1_missing_columns(self):
        """Test Layer 1 rejects missing required columns."""
        l1 = Layer1InputContract()
        df = pd.DataFrame({'foo': [1, 2, 3]})
        ok, result = l1.check(df)
        
        self.results.append(TestResult(
            "Layer 1: Rejects missing columns",
            ok == False and 'Missing required columns' in str(result.get('violations', [])),
            result.get('message', '')
        ))
        
    def _test_layer_1_valid_contract(self):
        """Test Layer 1 accepts valid data."""
        l1 = Layer1InputContract()
        df = self._create_valid_df()
        ok, result = l1.check(df)
        
        self.results.append(TestResult(
            "Layer 1: Accepts valid contract",
            ok == True and result['status'] == 'VALIDATED'
        ))
        
    def _test_layer_2_null_validation(self):
        """Test Layer 2 detects null values."""
        l2 = Layer2InputValidation()
        df = self._create_valid_df()
        df.loc[0, 'transaction_id'] = None  # Introduce null
        ok, result = l2.validate(df)
        
        has_critical = len(result.get('critical_issues', [])) > 0
        self.results.append(TestResult(
            "Layer 2: Detects null transaction_id",
            has_critical,
            f"Found {len(result.get('critical_issues', []))} critical issues"
        ))
        
    def _test_layer_2_duplicate_detection(self):
        """Test Layer 2 detects duplicates."""
        l2 = Layer2InputValidation()
        df = self._create_valid_df()
        df.loc[5] = df.loc[0]  # Create duplicate
        ok, result = l2.validate(df)
        
        has_duplicates = any('duplicate' in str(i).lower() for i in result.get('issues', []))
        self.results.append(TestResult(
            "Layer 2: Detects duplicate records",
            has_duplicates
        ))
    
    # =========================================================
    # LAYER 3 TESTS
    # =========================================================
    
    def _test_layer_3_feature_extraction(self):
        """Test Layer 3 creates expected features."""
        l3 = Layer3FeatureExtraction()
        df = self._create_valid_df()
        df_enriched, stats = l3.extract(df)
        
        expected_features = ['amount_zscore', 'amount_log', 'hour_of_day', 'is_weekend']
        has_all = all(f in df_enriched.columns for f in expected_features)
        
        self.results.append(TestResult(
            "Layer 3: Creates expected features",
            has_all and stats.total_features_created > 10,
            f"Created {stats.total_features_created} features"
        ))
        
    def _test_layer_3_edge_cases(self):
        """Test Layer 3 handles edge cases."""
        l3 = Layer3FeatureExtraction()
        
        # Single row DataFrame
        df = pd.DataFrame({
            'transaction_id': ['T1'],
            'amount': [100],
            'merchant_category': ['Retail'],
            'date': [datetime.now()]
        })
        
        try:
            df_enriched, stats = l3.extract(df)
            passed = 'amount_zscore' in df_enriched.columns
        except Exception as e:
            passed = False
            
        self.results.append(TestResult(
            "Layer 3: Handles single-row DataFrame",
            passed
        ))
    
    # =========================================================
    # LAYER 4 TESTS
    # =========================================================
    
    def _test_layer_4_structural_gate(self):
        """Test Layer 4.1 structural gate."""
        l4 = Layer4Inference()
        
        # Valid data should pass
        df = self._create_valid_df()
        ok, result = l4.sublayer_4_1_structural(df)
        
        # Empty data should fail
        empty_df = pd.DataFrame()
        ok_empty, _ = l4.sublayer_4_1_structural(empty_df)
        
        self.results.append(TestResult(
            "Layer 4.1: Structural gate works",
            ok == True and ok_empty == False
        ))
        
    def _test_layer_4_rules_scoring(self):
        """Test Layer 4.2 scoring logic."""
        l4 = Layer4Inference()
        df = self._create_valid_df()
        result = l4.sublayer_4_2_rules(df)
        
        has_dimensions = all(k in result['scores'] for k in ['Completeness', 'Accuracy', 'Validity'])
        score_valid = 0 <= result['composite'] <= 100
        
        self.results.append(TestResult(
            "Layer 4.2: Scoring produces valid output",
            has_dimensions and score_valid,
            f"Composite: {result['composite']:.1f}"
        ))
        
    def _test_layer_4_semantic_validation(self):
        """Test Layer 4.3 semantic validation."""
        l4 = Layer4Inference()
        
        # Create data with semantic violations (high amounts)
        df = self._create_valid_df()
        df['amount'] = [100000, 200000, 300000, 400000, 500000]  # All very high
        
        result = l4.sublayer_4_3_semantic(df)
        
        self.results.append(TestResult(
            "Layer 4.3: Detects semantic violations",
            result['critical_violations'] > 0,
            f"Found {result['critical_violations']} violations"
        ))
        
    def _test_layer_4_anomaly_detection(self):
        """Test Layer 4.4 anomaly detection."""
        l4 = Layer4Inference()
        
        # Create data with outliers (need enough normal data to establish stable mean/std)
        df = self._create_valid_df()
        # Add 15 normal rows
        normal_data = pd.DataFrame({
            'transaction_id': [f'N{i}' for i in range(15)],
            'amount': [100.0] * 15,
            'merchant_category': ['Retail'] * 15,
            'date': [datetime.now()] * 15
        })
        df = pd.concat([df, normal_data], ignore_index=True)
        # Add massive outlier
        df.loc[len(df)] = {
            'transaction_id': 'OUTLIER',
            'amount': 50000.0,
            'merchant_category': 'Retail',
            'date': datetime.now()
        }
        
        flags, metadata = l4.sublayer_4_4_anomaly(df)
        
        self.results.append(TestResult(
            "Layer 4.4: Detects anomalies",
            len(flags) > 0 and not metadata.get('degraded'),
            f"Found {len(flags)} anomalies"
        ))
        
    def _test_layer_4_ml_degradation(self):
        """Test Layer 4.4 safe degradation on ML failure."""
        l4 = Layer4Inference()
        df = self._create_valid_df()
        
        flags, metadata = l4.sublayer_4_4_anomaly(df, simulate_failure=True)
        
        self.results.append(TestResult(
            "Layer 4.4: Degrades safely on ML failure",
            len(flags) == 0 and metadata.get('degraded') == True
        ))
    
    # =========================================================
    # LAYER 5 & 6 TESTS
    # =========================================================
    
    def _test_layer_6_consistency_checks(self):
        """Test Layer 6 consistency detection."""
        l6 = Layer6Stability()
        
        # Create inconsistent scenario: High DQS but critical violations
        level, checks = l6.check_consistency(
            dqs=95.0,
            dimension_scores={'Completeness': 100, 'Accuracy': 90},
            critical_violations=5,
            anomaly_count=0
        )
        
        has_contradiction = level in [ConsistencyLevel.CONTRADICTION, ConsistencyLevel.WARNING]
        
        self.results.append(TestResult(
            "Layer 6: Detects score/violation contradiction",
            has_contradiction,
            level.value
        ))
    
    # =========================================================
    # LAYER 7 & 8 TESTS
    # =========================================================
    
    def _test_layer_7_conflict_detection(self):
        """Test Layer 7 conflict resolution."""
        l7 = Layer7ConflictDetection()
        
        # Create conflict: High rules score but many ML anomalies
        result = l7.resolve(
            rule_dqs=95.0,
            dimension_scores={'Completeness': 100, 'Accuracy': 90},
            ml_anomaly_count=20,
            ml_high_severity_count=5,
            semantic_violations=0,
            semantic_critical=False
        )
        
        self.results.append(TestResult(
            "Layer 7: Detects Rules vs ML conflict",
            result['total_conflicts'] > 0 and result.get('rules_authority_applied'),
            f"{result['total_conflicts']} conflicts detected"
        ))
        
    def _test_layer_8_confidence_calculation(self):
        """Test Layer 8 confidence scoring."""
        l8 = Layer8ConfidenceBand()
        
        # High quality scenario
        result = l8.calculate(
            dqs=95.0,
            dimension_scores={'Completeness': 100, 'Accuracy': 90},
            anomaly_count=2,
            conflict_result={'conflicts': [], 'total_conflicts': 0},
            row_count=100
        )
        
        self.results.append(TestResult(
            "Layer 8: Calculates confidence band",
            result['band'] == 'HIGH' and result['score'] > 80,
            f"Band: {result['band']}, Score: {result['score']}"
        ))
    
    # =========================================================
    # LAYER 9 TESTS
    # =========================================================
    
    def _test_layer_9_decision_safe(self):
        """Test Layer 9 SAFE_TO_USE decision."""
        l9 = Layer9DecisionGate()
        
        result = l9.decide(
            dqs_composite=90.0,
            dimension_scores={'Completeness': 95, 'Accuracy': 85},
            anomaly_flags=[],
            confidence_band="HIGH",
            conflict_result={'conflicts': [], 'escalation_required': False}
        )
        
        self.results.append(TestResult(
            "Layer 9: Issues SAFE_TO_USE for clean data",
            result.action == DecisionAction.SAFE_TO_USE
        ))
        
    def _test_layer_9_decision_escalate(self):
        """Test Layer 9 ESCALATE decision."""
        l9 = Layer9DecisionGate()
        
        result = l9.decide(
            dqs_composite=25.0,  # Very low
            dimension_scores={'Completeness': 20, 'Accuracy': 30},
            anomaly_flags=[],
            confidence_band="LOW",
            conflict_result={'conflicts': [], 'escalation_required': False}
        )
        
        self.results.append(TestResult(
            "Layer 9: Issues ESCALATE for critical failure",
            result.action == DecisionAction.ESCALATE
        ))
    
    # =========================================================
    # LAYER 10 TESTS
    # =========================================================
    
    def _test_layer_10_responsibility(self):
        """Test Layer 10 responsibility assignment."""
        l10 = Layer10Responsibility()
        
        handoff = l10.get_handoff("ESCALATE", {"dqs": 30})
        
        has_accountability = 'accountability' in handoff
        has_sla = 'sla' in handoff
        has_liability = 'liability' in handoff
        
        self.results.append(TestResult(
            "Layer 10: Produces complete handoff",
            has_accountability and has_sla and has_liability
        ))
    
    # =========================================================
    # LAYER 11 TESTS
    # =========================================================
    
    def _test_layer_11_logging(self):
        """Test Layer 11 logging functionality."""
        logger = Layer11Logging()
        
        logger.log_event("Test Layer", "TEST", {"key": "value"})
        logger.log_event("Test Layer", "TEST", {"key": "value"}, LogSeverity.WARN)
        
        logs = logger.get_logs()
        summary = logger.get_summary()
        
        self.results.append(TestResult(
            "Layer 11: Logging works correctly",
            len(logs) == 2 and 'trace_id' in summary,
            f"Trace ID: {logger.trace_id}"
        ))
    
    # =========================================================
    # INTEGRATION TESTS
    # =========================================================
    
    def _test_full_pipeline_clean_data(self):
        """Test full pipeline with clean data."""
        df = self._create_valid_df()
        result = self.pipeline.analyze(df)
        
        self.results.append(TestResult(
            "Integration: Clean data produces valid result",
            result.composite_score > 70 and result.overall_status in ["SAFE_TO_USE", "REVIEW_REQUIRED"],
            f"Score: {result.composite_score}, Status: {result.overall_status}"
        ))
        
    def _test_full_pipeline_problematic_data(self):
        """Test full pipeline with problematic data."""
        df = self._create_problematic_df()
        result = self.pipeline.analyze(df)
        
        self.results.append(TestResult(
            "Integration: Problematic data triggers review/escalate",
            result.overall_status in ["REVIEW_REQUIRED", "ESCALATE"],
            f"Score: {result.composite_score}, Status: {result.overall_status}"
        ))
        
    def _test_full_pipeline_ml_failure(self):
        """Test full pipeline with simulated ML failure."""
        df = self._create_valid_df()
        result = self.pipeline.analyze(df, simulate_ml_failure=True)
        
        # Should still produce a result (safe degradation)
        self.results.append(TestResult(
            "Integration: ML failure handled gracefully",
            result.composite_score >= 0 and result.overall_status != "NO_ACTION",
            f"Status: {result.overall_status}, Confidence: {result.confidence_band}"
        ))
    
    # =========================================================
    # HELPER METHODS
    # =========================================================
    
    def _create_valid_df(self) -> pd.DataFrame:
        """Create a valid test DataFrame."""
        return pd.DataFrame({
            'transaction_id': ['T001', 'T002', 'T003', 'T004', 'T005'],
            'amount': [100.0, 250.0, 75.0, 500.0, 150.0],
            'merchant_category': ['Retail', 'Food', 'Transport', 'Retail', 'Food'],
            'date': pd.date_range(start='2024-01-01', periods=5, freq='D'),
            'currency': ['USD', 'USD', 'EUR', 'USD', 'GBP'],
            'status': ['COMPLETED', 'COMPLETED', 'PENDING', 'COMPLETED', 'COMPLETED']
        })
        
    def _create_problematic_df(self) -> pd.DataFrame:
        """Create a DataFrame with quality issues."""
        df = pd.DataFrame({
            'transaction_id': ['T001', 'T001', 'T003', None, 'T005'],  # Duplicates and null
            'amount': [100.0, -50.0, 75.0, 1000000.0, 150.0],  # Negative and extreme
            'merchant_category': ['TESTXXX', 'Food', None, 'Retail', 'Food'],  # Placeholder and null
            'date': ['2024-01-01', 'invalid', '2024-01-03', '2030-01-01', '2024-01-05'],  # Invalid and future
            'currency': ['USD', 'XXX', 'EUR', 'USD', 'GBP'],  # Invalid currency
            'status': ['COMPLETED', 'COMPLETED', 'PENDING', 'FAILED', 'COMPLETED']
        })
        return df
        
    def _print_summary(self):
        """Print test summary."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        
        print("\n" + "="*60)
        print("[SUMMARY] TEST RESULTS")
        print("="*60)
        
        for result in self.results:
            print(result)
            
        print("\n" + "-"*60)
        print(f"Total: {len(self.results)} | Passed: {passed} | Failed: {failed}")
        print(f"Success Rate: {passed/len(self.results)*100:.1f}%")
        print("="*60 + "\n")


def run_tests():
    """Run the test suite."""
    suite = TestSuite()
    results = suite.run_all_tests()
    
    # Return exit code based on results
    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
