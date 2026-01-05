"""
LAYERS 1 & 2: INPUT CONTRACT & VALIDATION
Purpose: Define and validate the data contract strictly.
Type: 100% Deterministic

Features:
- Configurable schema with data types
- Multi-level validation (schema, structural, semantic)
- Detailed error reporting
- Support for optional vs required fields
"""
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

class FieldType(Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    DATE = "date"
    BOOLEAN = "boolean"
    CATEGORY = "category"

@dataclass
class FieldContract:
    """Contract specification for a single field."""
    name: str
    field_type: FieldType
    required: bool = True
    nullable: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None  # Regex pattern for strings

@dataclass
class DataContract:
    """Complete data contract specification."""
    name: str
    version: str
    fields: List[FieldContract]
    min_rows: int = 1
    max_rows: Optional[int] = None
    
# Default Payment Transaction Contract
PAYMENT_CONTRACT = DataContract(
    name="PaymentTransactionContract",
    version="1.0.0",
    fields=[
        FieldContract("transaction_id", FieldType.STRING, required=True, nullable=False),
        FieldContract("amount", FieldType.FLOAT, required=True, nullable=False, min_value=0),
        FieldContract("merchant_category", FieldType.CATEGORY, required=True, nullable=False),
        FieldContract("date", FieldType.DATE, required=True, nullable=False),
        FieldContract("currency", FieldType.CATEGORY, required=False, nullable=True, 
                     allowed_values=["USD", "EUR", "GBP", "INR", "JPY"]),
        FieldContract("status", FieldType.CATEGORY, required=False, nullable=True,
                     allowed_values=["COMPLETED", "PENDING", "FAILED", "REFUNDED"]),
    ],
    min_rows=5
)

class Layer1InputContract:
    """
    LAYER 1: INPUT CONTRACT LAYER
    Purpose: Define EXACTLY what data we accept.
    Type: 100% Deterministic
    
    Validates:
    - Required columns exist
    - Data types are compatible
    - File is not empty
    """
    def __init__(self, contract: DataContract = PAYMENT_CONTRACT):
        self.contract = contract
        
    def check(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate DataFrame against contract.
        Returns: (success, details_dict)
        """
        violations = []
        warnings = []
        
        # Check 1: Empty DataFrame
        if df.empty:
            return False, {
                "status": "REJECTED",
                "code": "EMPTY_FILE",
                "message": "Input file contains no data",
                "violations": ["File is empty"]
            }
        
        # Check 2: Minimum rows
        if len(df) < self.contract.min_rows:
            violations.append(f"Row count ({len(df)}) below minimum ({self.contract.min_rows})")
            
        # Check 3: Maximum rows
        if self.contract.max_rows and len(df) > self.contract.max_rows:
            warnings.append(f"Row count ({len(df)}) exceeds recommended maximum ({self.contract.max_rows})")
        
        # Check 4: Required columns
        required_fields = [f.name for f in self.contract.fields if f.required]
        missing_required = [f for f in required_fields if f not in df.columns]
        
        if missing_required:
            violations.append(f"Missing required columns: {missing_required}")
            
        # Check 5: Data type compatibility (for existing columns)
        type_mismatches = []
        for field in self.contract.fields:
            if field.name in df.columns:
                if not self._check_dtype_compatible(df[field.name], field.field_type):
                    type_mismatches.append(f"{field.name} (expected {field.field_type.value})")
                    
        if type_mismatches:
            warnings.append(f"Potential type mismatches: {type_mismatches}")
            
        # Build result
        if violations:
            return False, {
                "status": "REJECTED",
                "code": "CONTRACT_VIOLATION",
                "message": f"{len(violations)} contract violation(s) found",
                "violations": violations,
                "warnings": warnings,
                "contract_version": self.contract.version
            }
            
        return True, {
            "status": "VALIDATED",
            "code": "CONTRACT_PASSED",
            "message": "Input contract validated successfully",
            "columns_found": len(df.columns),
            "rows_found": len(df),
            "warnings": warnings,
            "contract_version": self.contract.version
        }
    
    def _check_dtype_compatible(self, series: pd.Series, expected_type: FieldType) -> bool:
        """Check if pandas series is compatible with expected type."""
        if expected_type == FieldType.STRING:
            return series.dtype == object or pd.api.types.is_string_dtype(series)
        elif expected_type == FieldType.INTEGER:
            return pd.api.types.is_integer_dtype(series)
        elif expected_type == FieldType.FLOAT:
            return pd.api.types.is_numeric_dtype(series)
        elif expected_type == FieldType.DATE:
            return pd.api.types.is_datetime64_any_dtype(series) or series.dtype == object
        elif expected_type == FieldType.BOOLEAN:
            return pd.api.types.is_bool_dtype(series)
        elif expected_type == FieldType.CATEGORY:
            return series.dtype == object or pd.api.types.is_categorical_dtype(series)
        return True


class Layer2InputValidation:
    """
    LAYER 2: INPUT VALIDATION LAYER
    Purpose: Verify structural integrity and deep schema compliance.
    Type: 100% Deterministic
    
    Validates:
    - Data integrity (null patterns, duplicates)
    - Value ranges
    - Cross-field consistency
    """
    def __init__(self, contract: DataContract = PAYMENT_CONTRACT):
        self.contract = contract
        
    def validate(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Deep validation of DataFrame structure and values.
        Returns: (success, details_dict)
        """
        issues = []
        warnings = []
        field_validations = {}
        
        for field in self.contract.fields:
            if field.name not in df.columns:
                continue
                
            col = df[field.name]
            field_issues = []
            
            # Null check
            null_count = col.isnull().sum()
            null_pct = (null_count / len(df)) * 100
            
            if not field.nullable and null_count > 0:
                field_issues.append(f"NULL values found: {null_count} ({null_pct:.1f}%)")
                if field.required:
                    issues.append(f"CRITICAL: Required field '{field.name}' has NULL values")
                    
            # Range validation for numeric fields
            if field.field_type in [FieldType.INTEGER, FieldType.FLOAT]:
                numeric_col = pd.to_numeric(col, errors='coerce')
                
                if field.min_value is not None:
                    below_min = (numeric_col < field.min_value).sum()
                    if below_min > 0:
                        field_issues.append(f"Values below minimum ({field.min_value}): {below_min}")
                        issues.append(f"Field '{field.name}' has {below_min} values below minimum")
                        
                if field.max_value is not None:
                    above_max = (numeric_col > field.max_value).sum()
                    if above_max > 0:
                        field_issues.append(f"Values above maximum ({field.max_value}): {above_max}")
                        warnings.append(f"Field '{field.name}' has {above_max} values above maximum")
                        
            # Allowed values check for categories
            if field.allowed_values and field.field_type == FieldType.CATEGORY:
                invalid_values = col[~col.isin(field.allowed_values) & col.notna()]
                if len(invalid_values) > 0:
                    unique_invalid = invalid_values.unique()[:5]  # Show first 5
                    field_issues.append(f"Invalid values found: {list(unique_invalid)}")
                    warnings.append(f"Field '{field.name}' has {len(invalid_values)} invalid categorical values")
                    
            field_validations[field.name] = {
                "null_count": int(null_count),
                "null_percentage": round(null_pct, 2),
                "issues": field_issues,
                "status": "FAIL" if field_issues else "PASS"
            }
        
        # Check for duplicate primary keys (transaction_id)
        if 'transaction_id' in df.columns:
            dupe_count = df['transaction_id'].duplicated().sum()
            if dupe_count > 0:
                issues.append(f"CRITICAL: {dupe_count} duplicate transaction_id values found")
                field_validations['transaction_id']['duplicates'] = int(dupe_count)
        
        # ADVERSARIAL INPUT DETECTION
        adversarial_patterns = [
            # SQL Injection patterns
            r"(?i)(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|OR\s+1\s*=\s*1)",
            # Script injection
            r"<script.*?>|javascript:|onclick|onerror",
            # Path traversal
            r"\.\./|\.\.\\",
            # Shell commands
            r"(?i)(;|\||`|\$\(|&&)\s*(cat|ls|rm|wget|curl|bash|sh|exec)",
        ]
        import re
        adversarial_flags = []
        
        for col in df.select_dtypes(include=['object']).columns:
            for idx, val in df[col].items():
                if pd.isna(val):
                    continue
                val_str = str(val)
                for pattern in adversarial_patterns:
                    if re.search(pattern, val_str):
                        adversarial_flags.append({
                            "row": idx,
                            "column": col,
                            "pattern": pattern[:30],
                            "value_snippet": val_str[:50]
                        })
                        break  # One flag per cell is enough
                        
        if adversarial_flags:
            issues.append(f"CRITICAL: {len(adversarial_flags)} potential adversarial inputs detected")
            field_validations['_adversarial'] = {
                "count": len(adversarial_flags),
                "samples": adversarial_flags[:5],  # Show first 5
                "status": "FAIL"
            }
                
        # Calculate overall validation score
        total_fields = len(field_validations)
        passed_fields = sum(1 for v in field_validations.values() if v.get('status') == 'PASS')
        validation_score = (passed_fields / total_fields * 100) if total_fields > 0 else 0
        
        has_critical = any("CRITICAL" in issue for issue in issues)
        has_adversarial = len(adversarial_flags) > 0
        
        return (not has_critical, {
            "status": "VALIDATED" if not has_critical else "FAILED",
            "validation_score": round(validation_score, 1),
            "fields_validated": total_fields,
            "fields_passed": passed_fields,
            "critical_issues": [i for i in issues if "CRITICAL" in i],
            "issues": issues,
            "warnings": warnings,
            "field_validations": field_validations,
            "adversarial_detected": has_adversarial,
            "adversarial_count": len(adversarial_flags)
        })

