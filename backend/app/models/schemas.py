"""Pydantic schemas for PRISM API."""
from typing import Any, Literal
from datetime import datetime

from pydantic import BaseModel, Field


# ============== STUDY CONDITIONS ==============
# 3-condition study design for rigorous comparison
StudyCondition = Literal["interactive", "static", "minimal"]
# interactive: Full PRISM with all features
# static: Basic explanations without interactivity (control)
# minimal: Decision only, no explanations (baseline)


# ============== PRE-STUDY QUESTIONNAIRE ==============

class PreStudyQuestionnaire(BaseModel):
    """Pre-study demographic and background questionnaire."""
    # Demographics
    age_range: Literal["18-24", "25-34", "35-44", "45-54", "55-64", "65+"] | None = None
    gender: Literal["male", "female", "non-binary", "prefer_not_to_say"] | None = None
    education: Literal["high_school", "bachelors", "masters", "doctorate", "other"] | None = None
    
    # Domain expertise
    finance_experience: int = Field(default=1, ge=1, le=5, description="1=None, 5=Expert")
    credit_familiarity: int = Field(default=1, ge=1, le=5, description="Familiarity with credit decisions")
    
    # AI/ML familiarity
    ai_familiarity: int = Field(default=1, ge=1, le=5, description="1=Not at all, 5=Very familiar")
    ai_trust_general: int = Field(default=3, ge=1, le=5, description="General trust in AI systems")
    explainable_ai_exposure: bool = Field(default=False, description="Previous exposure to XAI")
    
    # Tech comfort
    tech_comfort: int = Field(default=3, ge=1, le=5, description="Comfort with technology")


# ============== POST-STUDY QUESTIONNAIRE ==============

class NasaTLX(BaseModel):
    """NASA Task Load Index - trimmed to 3 key cognitive load items."""
    mental_demand: int = Field(..., ge=1, le=21, description="How mentally demanding was understanding the explanations?")
    effort: int = Field(..., ge=1, le=21, description="How hard did you have to work to make decisions?")
    frustration: int = Field(..., ge=1, le=21, description="How frustrated or stressed did you feel?")


class TrustInAutomation(BaseModel):
    """Trust in Automation scale - trimmed to 4 key items for XAI evaluation."""
    trustworthy: int = Field(..., ge=1, le=7, description="I trust the decisions made by this system")
    understandable: int = Field(..., ge=1, le=7, description="The explanations helped me understand why decisions were made")
    reliable: int = Field(..., ge=1, le=7, description="The system seems reliable")
    confident: int = Field(..., ge=1, le=7, description="I felt confident interpreting the results")


class UsabilityQuick(BaseModel):
    """Quick usability assessment - 2 key items."""
    easy_to_use: int = Field(..., ge=1, le=5, description="The system was easy to use")
    would_use_again: int = Field(..., ge=1, le=5, description="I would use this system again")


class PostStudyQuestionnaire(BaseModel):
    """Trimmed post-study questionnaire - 9 items + 2 open-ended."""
    nasa_tlx: NasaTLX
    trust: TrustInAutomation
    usability: UsabilityQuick
    
    # Open-ended (optional)
    most_helpful_feature: str | None = Field(default=None, description="What was most helpful?")
    improvement_suggestions: str | None = Field(default=None, description="How could we improve?")


# ============== TASK-BASED EVALUATION ==============

class EvaluationTask(BaseModel):
    """A single evaluation task for the participant."""
    task_id: str
    task_type: Literal["decision", "comprehension", "counterfactual", "attention_check"]
    row_index: int  # Which data row to evaluate
    question: str
    options: list[str] | None = None  # For multiple choice
    correct_answer: str | None = None  # For scoring (not shown to participant)
    time_limit_seconds: int | None = None


class TaskResponse(BaseModel):
    """Participant's response to a task."""
    session_id: str
    task_id: str
    response: str
    confidence: int = Field(default=3, ge=1, le=5, description="How confident in this answer?")
    time_taken_seconds: float
    timestamp: str | None = None


class TaskResult(BaseModel):
    """Scored result of a task."""
    task_id: str
    task_type: str
    response: str
    correct_answer: str | None
    is_correct: bool | None
    time_taken_seconds: float
    confidence: int


# ============== ENHANCED INTERACTION TRACKING ==============

class RichInteraction(BaseModel):
    """Enhanced interaction tracking with detailed metrics."""
    session_id: str
    action: str
    timestamp: str | None = None
    
    # Basic details
    details: dict[str, Any] = Field(default_factory=dict)
    
    # Time tracking
    time_since_session_start_ms: int | None = None
    time_since_last_interaction_ms: int | None = None
    
    # Engagement metrics
    viewport_time_ms: int | None = Field(default=None, description="Time element was in viewport")
    hover_time_ms: int | None = Field(default=None, description="Time hovering over element")
    scroll_depth_percent: float | None = Field(default=None, ge=0, le=100)
    
    # Section-specific timing
    section_times: dict[str, int] | None = Field(default=None, description="Time spent per UI section in ms")
    
    # Decision tracking
    initial_decision: str | None = None
    final_decision: str | None = None
    decision_changed: bool = False


# ============== STUDY MANAGEMENT SCHEMAS ==============

class StudySessionCreate(BaseModel):
    """Create a new study session."""
    participant_id: str = Field(..., description="Unique participant identifier")
    condition: StudyCondition = Field(..., description="Study condition")
    pre_questionnaire: PreStudyQuestionnaire | None = Field(default=None, description="Pre-study questionnaire responses")


class StudySessionResponse(BaseModel):
    """Study session details."""
    session_id: str
    participant_id: str
    condition: str
    started_at: str
    ended_at: str | None = None
    is_active: bool = True
    total_interactions: int = 0
    total_decisions: int = 0
    total_whatif_explorations: int = 0


class InteractionLog(BaseModel):
    """Log a user interaction for study analysis."""
    session_id: str
    action: str = Field(..., description="Action type: view_decision, change_mode, whatif_adjust, export, etc.")
    details: dict[str, Any] = Field(default_factory=dict, description="Action-specific details")
    timestamp: str | None = None  # Auto-filled if not provided


class StudyMetrics(BaseModel):
    """Aggregated metrics for a study session."""
    session_id: str
    participant_id: str
    condition: str
    duration_seconds: float
    total_interactions: int
    decisions_viewed: int
    whatif_explorations: int
    mode_switches: int
    exports: int
    time_to_first_decision_seconds: float | None
    avg_time_per_decision_seconds: float | None
    features_explored: list[str]
    feedback_submitted: bool
    
    # Enhanced metrics
    tasks_completed: int = 0
    tasks_correct: int = 0
    task_accuracy: float | None = None
    attention_checks_passed: int = 0
    attention_checks_total: int = 0
    avg_hover_time_ms: float | None = None
    avg_scroll_depth: float | None = None
    preferred_explanation_mode: str | None = None
    decision_reversals: int = 0


class StudyExportData(BaseModel):
    """Complete study data export for statistical analysis."""
    # Session summary
    sessions: list[dict[str, Any]]
    
    # Detailed interaction log
    interactions: list[dict[str, Any]]
    
    # Task results
    task_results: list[dict[str, Any]]
    
    # Questionnaire responses
    pre_questionnaires: list[dict[str, Any]]
    post_questionnaires: list[dict[str, Any]]
    
    # Aggregate statistics
    summary: dict[str, Any]


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


class FeedbackCreate(BaseModel):
    """User feedback submission."""

    interpretability_rating: int | None = Field(None, ge=1, le=5)
    clarity_rating: int | None = Field(None, ge=1, le=5)
    usability_rating: int | None = Field(None, ge=1, le=5)
    trust_rating: int | None = Field(None, ge=1, le=5)
    cognitive_load_rating: int | None = Field(None, ge=1, le=5)
    comments: str | None = None
    session_id: str | None = None


# ============== STUDY CONFIGURATION ==============

class StudyConfig(BaseModel):
    """Configuration for a study run."""
    study_id: str = Field(default="prism_study_v1")
    conditions: list[StudyCondition] = Field(default=["interactive", "static", "minimal"])
    tasks_per_session: int = Field(default=10)
    include_attention_checks: bool = Field(default=True)
    attention_check_frequency: int = Field(default=3, description="Insert attention check every N tasks")
    randomize_task_order: bool = Field(default=True)
    random_seed: int = Field(default=42, description="For reproducibility")
