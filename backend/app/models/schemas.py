"""Pydantic schemas for PRISM API."""
from typing import Any

from pydantic import BaseModel, Field


class TrustCalibration(BaseModel):
    """Trust calibration data for a decision."""
    model_confidence: float
    historical_accuracy: float | None = Field(default=None, description="How accurate the model has been historically at this confidence level")
    calibration_warning: str | None = None
    complexity_score: float = Field(default=0.5, ge=0, le=1, description="Explanation complexity (0=simple, 1=complex)")
    estimated_read_time_seconds: int = Field(default=30, description="Estimated time to understand the explanation")


class DecisionResponse(BaseModel):
    """PRISM decision engine output: decision, confidence, probabilities."""

    decision: str = Field(..., description="+ (approved) or - (rejected)")
    confidence: float = Field(..., ge=0, le=1)
    probabilities: dict[str, float] = Field(default_factory=dict)


class ShapValuesResponse(BaseModel):
    """SHAP decision-factor contributions for one instance."""

    decision_factor_names: list[str]
    values: list[float]
    base_value: float
    decision: str


class CounterfactualResponse(BaseModel):
    """Counterfactual what-if result."""

    original: dict[str, Any]
    modified: dict[str, Any]
    original_decision: str
    new_decision: str
    changed_decision_factors: list[str]


class ExplanationLayerResponse(BaseModel):
    """Plain-language bullets, directional reasoning, decision-factor summary."""

    bullets: list[str]
    summary: list[dict[str, Any]]
    directional_reasoning: str


class UncertaintyResponse(BaseModel):
    """Confidence band, stability, warnings."""

    confidence_band: str
    stable: bool
    warning: str | None
    volatility_note: str | None


class CounterfactualPreviewItem(BaseModel):
    """Single minimal-change suggestion."""

    suggestion: str
    decision_factor: str
    change: float | None


class ExportFormat(BaseModel):
    """Export format selection."""

    format: str = Field(..., pattern="^(csv|pdf)$")


class ExportRequest(BaseModel):
    """Export request body."""

    format: str = Field("csv", pattern="^(csv|pdf)$")
    data: dict[str, Any] = Field(default_factory=dict)
