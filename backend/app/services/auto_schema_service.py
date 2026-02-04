"""Auto-detection of dataset schema for dynamic model training.

This service analyzes uploaded CSV data to infer:
- Column types (numeric vs categorical)
- Suitable target columns
- Human-readable feature labels
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ColumnInfo:
    """Information about a single column."""
    name: str
    dtype: str  # "numeric" or "categorical"
    unique_count: int
    null_count: int
    sample_values: list[Any]
    suggested_label: str
    is_target_candidate: bool = False
    target_score: float = 0.0


@dataclass
class SchemaAnalysis:
    """Result of schema analysis for a dataset."""
    columns: list[ColumnInfo]
    suggested_target: str | None
    numeric_cols: list[str]
    categorical_cols: list[str]
    row_count: int
    warnings: list[str] = field(default_factory=list)


class AutoSchemaService:
    """Auto-detect schema from uploaded data."""

    # Column name patterns that suggest categorical variables
    CATEGORICAL_PATTERNS = [
        r"^(is_|has_|was_|did_)",  # Boolean prefixes
        r"(type|status|category|class|group|level|grade)$",
        r"(gender|sex|race|country|state|city)",
        r"(yes|no|true|false)",
    ]
    
    # Column name patterns that suggest target variables
    TARGET_PATTERNS = [
        r"^(target|label|class|outcome|result|y)$",
        r"(approved|rejected|default|churn|attrition|fraud)",
        r"(positive|negative|success|failure)",
        r"(diagnosis|prediction|is_)",
    ]
    
    # Max unique values for a column to be considered categorical
    MAX_CATEGORICAL_UNIQUE = 20
    
    # Min unique values for a column to be considered a valid target
    MIN_TARGET_UNIQUE = 2
    MAX_TARGET_UNIQUE = 10

    def analyze(self, df: pd.DataFrame) -> SchemaAnalysis:
        """Analyze a DataFrame and detect schema."""
        columns = []
        numeric_cols = []
        categorical_cols = []
        warnings = []
        
        for col in df.columns:
            col_info = self._analyze_column(df, col)
            columns.append(col_info)
            
            if col_info.dtype == "numeric":
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        
        # Find best target candidate
        target_candidates = [c for c in columns if c.is_target_candidate]
        target_candidates.sort(key=lambda x: x.target_score, reverse=True)
        
        suggested_target = target_candidates[0].name if target_candidates else None
        
        # Generate warnings
        if not suggested_target:
            warnings.append("No suitable target column found. Please select manually.")
        if len(df) < 50:
            warnings.append(f"Dataset has only {len(df)} rows. Consider using more data for reliable training.")
        if len(columns) > 100:
            warnings.append(f"Dataset has {len(columns)} columns. Training may be slow.")
        if df.isnull().sum().sum() > len(df) * len(df.columns) * 0.1:
            warnings.append("Dataset has >10% missing values. Results may vary.")
        
        return SchemaAnalysis(
            columns=columns,
            suggested_target=suggested_target,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            row_count=len(df),
            warnings=warnings,
        )

    def _analyze_column(self, df: pd.DataFrame, col: str) -> ColumnInfo:
        """Analyze a single column."""
        series = df[col]
        
        # Basic stats
        unique_count = series.nunique()
        null_count = series.isnull().sum()
        
        # Determine type
        dtype = self._infer_dtype(series, unique_count)
        
        # Sample values (non-null)
        non_null = series.dropna()
        sample_values = non_null.head(5).tolist() if len(non_null) > 0 else []
        
        # Generate human-readable label
        suggested_label = self._generate_label(col)
        
        # Check if target candidate
        is_target_candidate, target_score = self._check_target_candidate(
            col, series, unique_count, dtype
        )
        
        return ColumnInfo(
            name=col,
            dtype=dtype,
            unique_count=unique_count,
            null_count=null_count,
            sample_values=sample_values,
            suggested_label=suggested_label,
            is_target_candidate=is_target_candidate,
            target_score=target_score,
        )

    def _infer_dtype(self, series: pd.Series, unique_count: int) -> str:
        """Infer whether a column is numeric or categorical."""
        # Check pandas dtype
        if pd.api.types.is_numeric_dtype(series):
            # Even if numeric, could be categorical if few unique values
            if unique_count <= self.MAX_CATEGORICAL_UNIQUE:
                # Check if values look like codes (small integers)
                non_null = series.dropna()
                if len(non_null) > 0:
                    if all(isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer()) 
                           for v in non_null.head(100)):
                        # Small integers could be categorical codes
                        if non_null.max() <= 20 and non_null.min() >= 0:
                            return "categorical"
            return "numeric"
        
        # Check if can be converted to numeric
        try:
            pd.to_numeric(series.dropna().head(100))
            return "numeric"
        except (ValueError, TypeError):
            pass
        
        return "categorical"

    def _generate_label(self, col_name: str) -> str:
        """Generate a human-readable label from column name."""
        # Handle common abbreviations
        abbrev_map = {
            "num": "Number of",
            "amt": "Amount",
            "cnt": "Count",
            "pct": "Percent",
            "avg": "Average",
            "tot": "Total",
            "max": "Maximum",
            "min": "Minimum",
            "yr": "Year",
            "mo": "Month",
            "idx": "Index",
            "id": "ID",
        }
        
        # Split on underscores, camelCase, and numbers
        parts = re.split(r'[_\s]+|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', col_name)
        
        # Process parts
        processed = []
        for part in parts:
            part_lower = part.lower()
            if part_lower in abbrev_map:
                processed.append(abbrev_map[part_lower])
            elif part:
                processed.append(part.capitalize())
        
        return " ".join(processed) if processed else col_name

    def _check_target_candidate(
        self, col: str, series: pd.Series, unique_count: int, dtype: str
    ) -> tuple[bool, float]:
        """Check if a column is a suitable target variable."""
        score = 0.0
        
        # Must have right number of unique values
        if unique_count < self.MIN_TARGET_UNIQUE or unique_count > self.MAX_TARGET_UNIQUE:
            return False, 0.0
        
        # Binary is ideal
        if unique_count == 2:
            score += 5.0
        elif unique_count <= 5:
            score += 3.0
        else:
            score += 1.0
        
        # Check name patterns
        col_lower = col.lower()
        for pattern in self.TARGET_PATTERNS:
            if re.search(pattern, col_lower):
                score += 3.0
                break
        
        # Exact name matches get extra score
        if col_lower in ("target", "label", "class", "outcome", "y"):
            score += 5.0
        
        # Check for typical binary values
        non_null = series.dropna()
        if len(non_null) > 0:
            unique_vals = set(str(v).lower() for v in non_null.unique())
            
            # Common binary patterns
            binary_patterns = [
                {"0", "1"},
                {"yes", "no"},
                {"true", "false"},
                {"positive", "negative"},
                {"+", "-"},
                {"approved", "rejected"},
                {"pass", "fail"},
            ]
            
            for pattern in binary_patterns:
                if unique_vals == pattern or unique_vals.issubset(pattern):
                    score += 2.0
                    break
        
        # Prefer columns at the end (common convention)
        # This is handled by the caller based on position
        
        return score > 0, score

    def get_target_values(self, df: pd.DataFrame, target_col: str) -> dict[str, Any]:
        """Get unique values for a target column with suggested positive/negative labels."""
        if target_col not in df.columns:
            return {"error": f"Column '{target_col}' not found"}
        
        series = df[target_col]
        unique_vals = series.dropna().unique().tolist()
        
        if len(unique_vals) > 10:
            return {
                "error": f"Column has too many unique values ({len(unique_vals)}). Must be <= 10.",
                "unique_values": unique_vals[:10],
            }
        
        # Try to identify positive/negative values
        positive_val = None
        negative_val = None
        
        if len(unique_vals) == 2:
            v1, v2 = unique_vals
            v1_str, v2_str = str(v1).lower(), str(v2).lower()
            
            # Common mappings
            positive_indicators = ["1", "yes", "true", "positive", "+", "approved", "pass"]
            negative_indicators = ["0", "no", "false", "negative", "-", "rejected", "fail"]
            
            if v1_str in positive_indicators or v2_str in negative_indicators:
                positive_val, negative_val = v1, v2
            elif v2_str in positive_indicators or v1_str in negative_indicators:
                positive_val, negative_val = v2, v1
            else:
                # Default: larger value is positive
                try:
                    positive_val = max(v1, v2)
                    negative_val = min(v1, v2)
                except TypeError:
                    positive_val, negative_val = v1, v2
        
        return {
            "unique_values": unique_vals,
            "suggested_positive": positive_val,
            "suggested_negative": negative_val,
            "value_counts": series.value_counts().to_dict(),
        }

    def suggest_labels(self, df: pd.DataFrame, domain_hint: str | None = None) -> dict[str, str]:
        """Generate suggested labels for the positive/negative outcomes."""
        # Default labels by domain
        domain_labels = {
            "finance": ("Approved", "Rejected"),
            "credit": ("Approved", "Rejected"),
            "healthcare": ("Positive", "Negative"),
            "medical": ("Positive", "Negative"),
            "hr": ("Will Leave", "Will Stay"),
            "employment": ("Will Leave", "Will Stay"),
            "insurance": ("Will Claim", "Won't Claim"),
            "education": ("Pass", "Fail"),
            "legal": ("Guilty", "Not Guilty"),
        }
        
        if domain_hint:
            hint_lower = domain_hint.lower()
            for key, labels in domain_labels.items():
                if key in hint_lower:
                    return {"positive_label": labels[0], "negative_label": labels[1]}
        
        # Default
        return {"positive_label": "Positive", "negative_label": "Negative"}


# Singleton
auto_schema_service = AutoSchemaService()
