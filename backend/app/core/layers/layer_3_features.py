"""
LAYER 3: FEATURE / SIGNAL EXTRACTION LAYER
Purpose: Transform raw data into analyzable features.
Type: 100% Deterministic

Features:
- Statistical features (z-scores, percentiles, log transforms)
- Temporal features (hour, day, week patterns)
- Categorical encoding (hashing, frequency encoding)
- Cross-field derived features
- Anomaly detection prep features
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class FeatureStats:
    """Statistics about extracted features."""
    total_features_created: int
    statistical_features: int
    temporal_features: int
    categorical_features: int
    derived_features: int
    feature_names: List[str]

class Layer3FeatureExtraction:
    """
    LAYER 3: FEATURE EXTRACTION LAYER
    Purpose: Transform raw data into analyzable features.
    Type: 100% Deterministic
    
    Creates:
    - Statistical features for numeric columns
    - Temporal features from date columns
    - Encoded categorical features
    - Cross-field interaction features
    """
    def __init__(self):
        self._feature_log: List[str] = []
        self._stats: Dict[str, Any] = {}
        
    def extract(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, FeatureStats]:
        """
        Extract all features from DataFrame.
        Returns: (enriched_dataframe, feature_statistics)
        """
        features = df.copy()
        self._feature_log = []
        
        stat_count = 0
        temp_count = 0
        cat_count = 0
        derived_count = 0
        
        # ========================================
        # STATISTICAL FEATURES
        # ========================================
        if 'amount' in features.columns:
            # Convert to numeric, coercing errors
            features['amount'] = pd.to_numeric(features['amount'], errors='coerce')
            
            # Handle edge case: single row or all same values
            mean_amt = features['amount'].mean()
            std_amt = features['amount'].std()
            
            # Z-Score (handle zero std)
            if std_amt > 0:
                features['amount_zscore'] = (features['amount'] - mean_amt) / std_amt
            else:
                features['amount_zscore'] = 0.0
            self._feature_log.append('amount_zscore')
            stat_count += 1
            
            # Log transform (handle zero/negative)
            features['amount_log'] = np.log1p(features['amount'].clip(lower=0))
            self._feature_log.append('amount_log')
            stat_count += 1
            
            # Percentile rank
            features['amount_percentile'] = features['amount'].rank(pct=True) * 100
            self._feature_log.append('amount_percentile')
            stat_count += 1
            
            # Binned amount (quartiles)
            if len(features) >= 4:
                try:
                    features['amount_quartile'] = pd.qcut(
                        features['amount'], 
                        q=4, 
                        labels=['Q1_Low', 'Q2_MedLow', 'Q3_MedHigh', 'Q4_High'],
                        duplicates='drop'
                    )
                except ValueError:
                    features['amount_quartile'] = 'Unknown'
            else:
                features['amount_quartile'] = 'Unknown'
            self._feature_log.append('amount_quartile')
            stat_count += 1
            
            # IQR-based outlier flag
            q1 = features['amount'].quantile(0.25)
            q3 = features['amount'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            features['amount_iqr_outlier'] = (
                (features['amount'] < lower_bound) | 
                (features['amount'] > upper_bound)
            ).astype(int)
            self._feature_log.append('amount_iqr_outlier')
            stat_count += 1
            
            # Store stats for later use
            self._stats['amount'] = {
                'mean': mean_amt,
                'std': std_amt,
                'q1': q1,
                'q3': q3,
                'iqr': iqr
            }
            
        # ========================================
        # TEMPORAL FEATURES
        # ========================================
        if 'date' in features.columns:
            # Safely convert to datetime
            features['date'] = pd.to_datetime(features['date'], errors='coerce')
            
            # Only process rows with valid dates
            valid_dates = features['date'].notna()
            
            # Hour of day
            features['hour_of_day'] = features.loc[valid_dates, 'date'].dt.hour
            self._feature_log.append('hour_of_day')
            temp_count += 1
            
            # Day of week (0=Monday, 6=Sunday)
            features['day_of_week'] = features.loc[valid_dates, 'date'].dt.dayofweek
            self._feature_log.append('day_of_week')
            temp_count += 1
            
            # Weekend flag
            features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
            self._feature_log.append('is_weekend')
            temp_count += 1
            
            # Time of day bucket (Morning/Afternoon/Evening/Night)
            features['time_bucket'] = pd.cut(
                features['hour_of_day'],
                bins=[-1, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening']
            )
            self._feature_log.append('time_bucket')
            temp_count += 1
            
            # Day of month
            features['day_of_month'] = features.loc[valid_dates, 'date'].dt.day
            self._feature_log.append('day_of_month')
            temp_count += 1
            
            # Month
            features['month'] = features.loc[valid_dates, 'date'].dt.month
            self._feature_log.append('month')
            temp_count += 1
            
            # Days since epoch (for ML models)
            features['days_since_epoch'] = (
                features.loc[valid_dates, 'date'] - pd.Timestamp('1970-01-01')
            ).dt.days
            self._feature_log.append('days_since_epoch')
            temp_count += 1
            
            # Recency (days from max date in dataset)
            max_date = features['date'].max()
            features['days_from_latest'] = (max_date - features['date']).dt.days
            self._feature_log.append('days_from_latest')
            temp_count += 1
            
        # ========================================
        # CATEGORICAL FEATURES
        # ========================================
        if 'merchant_category' in features.columns:
            # Frequency encoding
            freq_map = features['merchant_category'].value_counts(normalize=True).to_dict()
            features['merchant_category_freq'] = features['merchant_category'].map(freq_map)
            self._feature_log.append('merchant_category_freq')
            cat_count += 1
            
            # Hash encoding (deterministic)
            features['merchant_category_hash'] = features['merchant_category'].apply(
                lambda x: hash(str(x)) % 10000
            )
            self._feature_log.append('merchant_category_hash')
            cat_count += 1
            
            # Count encoding
            count_map = features['merchant_category'].value_counts().to_dict()
            features['merchant_category_count'] = features['merchant_category'].map(count_map)
            self._feature_log.append('merchant_category_count')
            cat_count += 1
            
        if 'currency' in features.columns:
            # Currency volatility indicator (higher for exotic currencies)
            volatility_map = {
                'USD': 1.0, 'EUR': 1.1, 'GBP': 1.2, 
                'INR': 1.5, 'JPY': 1.3
            }
            features['currency_volatility'] = features['currency'].map(volatility_map).fillna(2.0)
            self._feature_log.append('currency_volatility')
            cat_count += 1
            
        # ========================================
        # DERIVED / CROSS-FIELD FEATURES
        # ========================================
        if 'amount' in features.columns and 'merchant_category' in features.columns:
            # Amount relative to category mean
            category_means = features.groupby('merchant_category')['amount'].transform('mean')
            features['amount_vs_category_mean'] = features['amount'] / category_means.replace(0, 1)
            self._feature_log.append('amount_vs_category_mean')
            derived_count += 1
            
            # High value for category flag
            category_75th = features.groupby('merchant_category')['amount'].transform(
                lambda x: x.quantile(0.75)
            )
            features['high_for_category'] = (features['amount'] > category_75th).astype(int)
            self._feature_log.append('high_for_category')
            derived_count += 1
            
        if 'is_weekend' in features.columns and 'amount' in features.columns:
            # Weekend spend pattern
            weekend_mean = features.loc[features['is_weekend'] == 1, 'amount'].mean()
            weekday_mean = features.loc[features['is_weekend'] == 0, 'amount'].mean()
            if pd.notna(weekend_mean) and pd.notna(weekday_mean) and weekday_mean > 0:
                features['weekend_spend_ratio'] = weekend_mean / weekday_mean
            else:
                features['weekend_spend_ratio'] = 1.0
            self._feature_log.append('weekend_spend_ratio')
            derived_count += 1
            
        # ========================================
        # ANOMALY DETECTION PREP
        # ========================================
        # Create a composite risk score based on multiple signals
        risk_components = []
        
        if 'amount_iqr_outlier' in features.columns:
            risk_components.append(features['amount_iqr_outlier'] * 25)
        if 'high_for_category' in features.columns:
            risk_components.append(features['high_for_category'] * 15)
        if 'is_weekend' in features.columns and 'hour_of_day' in features.columns:
            # Late night weekend transaction = higher risk
            late_night_weekend = (
                (features['is_weekend'] == 1) & 
                (features['hour_of_day'].isin([0, 1, 2, 3, 4, 5]))
            ).astype(int)
            risk_components.append(late_night_weekend * 20)
            
        if risk_components:
            features['composite_risk_score'] = sum(risk_components)
            features['composite_risk_score'] = features['composite_risk_score'].clip(0, 100)
            self._feature_log.append('composite_risk_score')
            derived_count += 1
            
        # Build stats object
        stats = FeatureStats(
            total_features_created=len(self._feature_log),
            statistical_features=stat_count,
            temporal_features=temp_count,
            categorical_features=cat_count,
            derived_features=derived_count,
            feature_names=self._feature_log.copy()
        )
        
        return features, stats
    
    def get_feature_importance(self, df_enriched: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate feature importance based on variance and correlation with amount.
        """
        importance = {}
        numeric_cols = df_enriched.select_dtypes(include=[np.number]).columns
        
        for col in self._feature_log:
            if col in numeric_cols:
                # Normalize variance as importance proxy
                variance = df_enriched[col].var()
                importance[col] = round(min(variance / 100, 1.0), 3)
                
        return importance


# ============================================================
# DIMENSION RELEVANCE ANALYZER
# ============================================================
@dataclass
class DimensionRelevance:
    """Result of dimension applicability check."""
    dimension: str
    applicable: bool
    reason: str
    priority: int  # 1=Critical, 2=High, 3=Medium
    required_columns: List[str]
    found_columns: List[str]


class DimensionRelevanceAnalyzer:
    """
    Automatically identifies which Data Quality dimensions are relevant
    for a given dataset based on schema analysis.
    
    This satisfies the requirement:
    "Automatically identify the relevant data quality dimensions to evaluate"
    """
    
    # Define what columns trigger each dimension
    DIMENSION_RULES = {
        'Completeness': {
            'description': 'Measures presence of required fields',
            'triggers': ['*'],  # Always applies
            'priority': 1,
        },
        'Accuracy': {
            'description': 'Measures correctness of values against known formats',
            'triggers': ['amount', 'currency', 'date', 'email', 'phone'],
            'priority': 1,
        },
        'Consistency': {
            'description': 'Measures uniformity across related fields',
            'triggers': ['currency', 'country', 'status', 'type'],
            'priority': 2,
        },
        'Timeliness': {
            'description': 'Measures freshness and temporal validity',
            'triggers': ['date', 'timestamp', 'created_at', 'updated_at', 'transaction_date'],
            'priority': 2,
        },
        'Uniqueness': {
            'description': 'Measures absence of duplicates in key fields',
            'triggers': ['transaction_id', 'id', 'reference', 'order_id', 'payment_id'],
            'priority': 1,
        },
        'Validity': {
            'description': 'Measures conformance to defined value sets',
            'triggers': ['status', 'type', 'category', 'merchant_category', 'payment_method'],
            'priority': 2,
        },
        'Integrity': {
            'description': 'Measures cross-reference consistency',
            'triggers': ['customer_id', 'merchant_id', 'account_id', 'user_id'],
            'priority': 3,
        },
    }
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, DimensionRelevance]:
        """
        Analyze DataFrame and return relevance for each dimension.
        
        Returns:
            Dict mapping dimension name to DimensionRelevance object
        """
        columns = [col.lower() for col in df.columns]
        original_columns = list(df.columns)
        results = {}
        
        for dimension, rules in self.DIMENSION_RULES.items():
            triggers = rules['triggers']
            found = []
            
            if triggers == ['*']:
                # Always applicable
                applicable = True
                found = original_columns[:3]  # Show first 3 as examples
                reason = "Applies to all datasets (measures null/missing values)"
            else:
                # Check if any trigger column exists
                for trigger in triggers:
                    for i, col in enumerate(columns):
                        if trigger in col:
                            found.append(original_columns[i])
                
                applicable = len(found) > 0
                if applicable:
                    reason = f"Relevant columns found: {', '.join(found[:3])}"
                else:
                    reason = f"No relevant columns (needs: {', '.join(triggers[:3])}...)"
            
            results[dimension] = DimensionRelevance(
                dimension=dimension,
                applicable=applicable,
                reason=reason,
                priority=rules['priority'],
                required_columns=triggers if triggers != ['*'] else ['any'],
                found_columns=found
            )
        
        return results
    
    def get_applicable_dimensions(self, df: pd.DataFrame) -> List[str]:
        """Return list of applicable dimension names."""
        relevance = self.analyze(df)
        return [dim for dim, rel in relevance.items() if rel.applicable]
    
    def get_dimension_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary suitable for display in the UI.
        """
        relevance = self.analyze(df)
        
        applicable = [r for r in relevance.values() if r.applicable]
        not_applicable = [r for r in relevance.values() if not r.applicable]
        
        return {
            "total_dimensions": len(relevance),
            "applicable_count": len(applicable),
            "not_applicable_count": len(not_applicable),
            "applicable": [
                {
                    "dimension": r.dimension,
                    "reason": r.reason,
                    "priority": r.priority,
                    "columns": r.found_columns[:3]
                }
                for r in sorted(applicable, key=lambda x: x.priority)
            ],
            "skipped": [
                {
                    "dimension": r.dimension,
                    "reason": r.reason
                }
                for r in not_applicable
            ],
            "schema_columns": list(df.columns),
            "auto_detected": True
        }

