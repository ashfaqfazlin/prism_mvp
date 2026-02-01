"""Domain configuration for multi-dataset PRISM support.

Each domain defines:
- Dataset schema (features, target, types)
- Human-readable labels for explainability
- Model paths
- Decision labels (what + and - mean)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import settings


@dataclass
class DomainConfig:
    """Configuration for a specific domain/dataset."""
    
    # Identifiers
    id: str
    name: str
    description: str
    
    # Schema
    feature_cols: list[str]
    target_col: str
    categorical_cols: list[str]
    numeric_cols: list[str]
    
    # Human-readable labels for features
    feature_labels: dict[str, str] = field(default_factory=dict)
    
    # Decision labels
    positive_label: str = "Approved"
    negative_label: str = "Rejected"
    positive_value: Any = 1
    negative_value: Any = 0
    
    # Paths (relative to artifacts dir)
    model_filename: str = "model.joblib"
    preprocessing_filename: str = "preprocessing.joblib"
    
    @property
    def model_path(self) -> Path:
        return settings.artifacts_dir / self.id / self.model_filename
    
    @property
    def preprocessing_path(self) -> Path:
        return settings.artifacts_dir / self.id / self.preprocessing_filename
    
    @property
    def artifacts_dir(self) -> Path:
        return settings.artifacts_dir / self.id


# ============== DOMAIN DEFINITIONS ==============

UCI_CREDIT_APPROVAL = DomainConfig(
    id="uci_credit_approval",
    name="UCI Credit Approval",
    description="Credit card application approval prediction",
    feature_cols=[f"A{i}" for i in range(1, 16)],
    target_col="A16",
    categorical_cols=["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"],
    numeric_cols=["A2", "A3", "A8", "A11", "A14", "A15"],
    feature_labels={
        "A1": "Gender",
        "A2": "Age",
        "A3": "Debt",
        "A4": "Marital Status",
        "A5": "Bank Customer",
        "A6": "Education Level",
        "A7": "Ethnicity",
        "A8": "Years Employed",
        "A9": "Prior Default",
        "A10": "Employment Status",
        "A11": "Credit Score",
        "A12": "Drivers License",
        "A13": "Citizenship",
        "A14": "Zip Code",
        "A15": "Income",
    },
    positive_label="Approved",
    negative_label="Rejected",
    positive_value="+",
    negative_value="-",
)

GERMAN_CREDIT = DomainConfig(
    id="german_credit",
    name="German Credit (Statlog)",
    description="Credit risk classification for loan applicants",
    feature_cols=[
        "checking_status", "duration", "credit_history", "purpose", "credit_amount",
        "savings_status", "employment", "installment_commitment", "personal_status",
        "other_parties", "residence_since", "property_magnitude", "age",
        "other_payment_plans", "housing", "existing_credits", "job",
        "num_dependents", "own_telephone", "foreign_worker"
    ],
    target_col="class",
    categorical_cols=[
        "checking_status", "credit_history", "purpose", "savings_status",
        "employment", "personal_status", "other_parties", "property_magnitude",
        "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker"
    ],
    numeric_cols=[
        "duration", "credit_amount", "installment_commitment", "residence_since",
        "age", "existing_credits", "num_dependents"
    ],
    feature_labels={
        "checking_status": "Checking Account Status",
        "duration": "Loan Duration (months)",
        "credit_history": "Credit History",
        "purpose": "Loan Purpose",
        "credit_amount": "Credit Amount",
        "savings_status": "Savings Account Status",
        "employment": "Employment Duration",
        "installment_commitment": "Installment Rate (%)",
        "personal_status": "Personal Status & Gender",
        "other_parties": "Other Debtors/Guarantors",
        "residence_since": "Years at Residence",
        "property_magnitude": "Property Type",
        "age": "Age (years)",
        "other_payment_plans": "Other Payment Plans",
        "housing": "Housing Type",
        "existing_credits": "Number of Existing Credits",
        "job": "Job Type",
        "num_dependents": "Number of Dependents",
        "own_telephone": "Has Telephone",
        "foreign_worker": "Foreign Worker",
    },
    positive_label="Good Credit",
    negative_label="Bad Credit",
    positive_value=1,
    negative_value=2,
)

TAIWAN_CREDIT_CARD = DomainConfig(
    id="taiwan_credit_card",
    name="Taiwan Credit Card Default",
    description="Credit card default prediction for next month",
    feature_cols=[
        "limit_bal", "sex", "education", "marriage", "age",
        "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
        "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6",
        "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"
    ],
    target_col="default_payment_next_month",
    categorical_cols=["sex", "education", "marriage"],
    numeric_cols=[
        "limit_bal", "age",
        "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
        "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6",
        "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"
    ],
    feature_labels={
        "limit_bal": "Credit Limit",
        "sex": "Gender",
        "education": "Education Level",
        "marriage": "Marital Status",
        "age": "Age",
        "pay_0": "Payment Status (Sep)",
        "pay_2": "Payment Status (Aug)",
        "pay_3": "Payment Status (Jul)",
        "pay_4": "Payment Status (Jun)",
        "pay_5": "Payment Status (May)",
        "pay_6": "Payment Status (Apr)",
        "bill_amt1": "Bill Amount (Sep)",
        "bill_amt2": "Bill Amount (Aug)",
        "bill_amt3": "Bill Amount (Jul)",
        "bill_amt4": "Bill Amount (Jun)",
        "bill_amt5": "Bill Amount (May)",
        "bill_amt6": "Bill Amount (Apr)",
        "pay_amt1": "Payment Amount (Sep)",
        "pay_amt2": "Payment Amount (Aug)",
        "pay_amt3": "Payment Amount (Jul)",
        "pay_amt4": "Payment Amount (Jun)",
        "pay_amt5": "Payment Amount (May)",
        "pay_amt6": "Payment Amount (Apr)",
    },
    positive_label="Will Default",
    negative_label="No Default",
    positive_value=1,
    negative_value=0,
)


# ============== DOMAIN REGISTRY ==============

DOMAIN_REGISTRY: dict[str, DomainConfig] = {
    "uci_credit_approval": UCI_CREDIT_APPROVAL,
    "german_credit": GERMAN_CREDIT,
    "taiwan_credit_card": TAIWAN_CREDIT_CARD,
}


def get_domain(domain_id: str) -> DomainConfig | None:
    """Get domain configuration by ID."""
    return DOMAIN_REGISTRY.get(domain_id)


def list_domains() -> list[DomainConfig]:
    """List all available domains."""
    return list(DOMAIN_REGISTRY.values())


def is_model_trained(domain_id: str) -> bool:
    """Check if a model has been trained for this domain."""
    domain = get_domain(domain_id)
    if not domain:
        return False
    return domain.model_path.exists() and domain.preprocessing_path.exists()
