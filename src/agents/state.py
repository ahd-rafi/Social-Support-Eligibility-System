"""LangGraph state schema for the application processing pipeline."""

from typing import TypedDict, List, Dict, Any, Optional


class ApplicationState(TypedDict):
    """State schema for the LangGraph agent pipeline.
    
    This state flows through all agents in the graph:
    START → extraction → validation → eligibility → decision → END
    """
    
    # Application Identity
    application_id: str
    applicant_id: Optional[str]
    application_ref: str
    chat_history: List[Dict[str, str]]  # Conversation messages for UI
    
    # Document Tracking
    uploaded_documents: List[Dict[str, Any]]  # [{type, filename, storage_path, size}]
    extraction_results: Dict[str, Dict[str, Any]]  # {doc_type: {extracted_data}}
    extraction_complete: bool
    
    # Validation
    validation_issues: List[Dict[str, Any]]  # [{field, source_a, source_b, severity, detail}]
    validation_passed: bool
    validation_complete: bool
    
    # Eligibility Scoring
    feature_vector: Dict[str, Any]  # {feature_name: value} - classifier input
    eligibility_score: float  # 0.0 – 1.0 confidence
    eligibility_category: str  # "approved_financial", "approved_enablement", "soft_decline"
    feature_importances: Dict[str, float]  # {feature_name: importance} - explainability
    eligibility_complete: bool
    
    # Recommendation
    recommendation: Dict[str, Any]  # {decision, reasoning, confidence, review_notes}
    enablement_programs: List[Dict[str, Any]]  # [{program_name, match_score, description}]
    recommendation_complete: bool
    
    # Observability & Status
    langfuse_trace_id: Optional[str]
    current_agent: str
    processing_status: str  # "pending", "extracting", "validating", "scoring", "recommending", "complete", "error"
    error: Optional[str]
    
    # Metadata
    started_at: Optional[str]
    completed_at: Optional[str]


def create_initial_state(application_id: str, application_ref: str) -> ApplicationState:
    """Create initial state for a new application.
    
    Args:
        application_id: Unique application identifier
        application_ref: Human-readable application reference (e.g., APP-001)
        
    Returns:
        Initial ApplicationState
    """
    return ApplicationState(
        application_id=application_id,
        applicant_id=None,
        application_ref=application_ref,
        chat_history=[],
        uploaded_documents=[],
        extraction_results={},
        extraction_complete=False,
        validation_issues=[],
        validation_passed=False,
        validation_complete=False,
        feature_vector={},
        eligibility_score=0.0,
        eligibility_category="",
        feature_importances={},
        eligibility_complete=False,
        recommendation={},
        enablement_programs=[],
        recommendation_complete=False,
        langfuse_trace_id=None,
        current_agent="",
        processing_status="pending",
        error=None,
        started_at=None,
        completed_at=None,
    )
