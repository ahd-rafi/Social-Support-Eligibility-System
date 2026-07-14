"""Pydantic schemas for API request/response models."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


# Request Models

class CreateApplicationRequest(BaseModel):
    """Request to create a new application."""
    applicant_name: Optional[str] = None
    household_size: Optional[int] = None
    num_dependents: Optional[int] = None
    monthly_income: Optional[float] = None


class UploadDocumentRequest(BaseModel):
    """Request to upload a document (metadata only, file via multipart)."""
    document_type: str = Field(..., description="Type of document: emirates_id, bank_statement, resume, credit_report, assets_liabilities")


class ChatMessageRequest(BaseModel):
    """Request to send a chat message."""
    message: str = Field(..., min_length=1, max_length=5000)


# Response Models

class ApplicationResponse(BaseModel):
    """Response with application details."""
    application_id: str
    application_ref: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None


class UploadDocumentResponse(BaseModel):
    """Response after document upload."""
    success: bool
    message: str
    document_count: int


class DocumentInfo(BaseModel):
    """Information about an uploaded document."""
    file_name: str
    file_type: str
    document_type: str
    uploaded_at: datetime


class ValidationIssue(BaseModel):
    """Validation issue details."""
    issue_type: str
    severity: str
    field_name: Optional[str] = None
    source_doc_a: Optional[str] = None
    source_doc_b: Optional[str] = None
    value_a: Optional[str] = None
    value_b: Optional[str] = None
    description: str


class EnablementProgram(BaseModel):
    """Enablement program recommendation."""
    program_name: str
    match_score: float
    description: str
    provider: Optional[str] = None
    duration: Optional[str] = None
    category: Optional[str] = None


class Recommendation(BaseModel):
    """Final recommendation details."""
    decision: str
    confidence: float
    reasoning: str
    key_factors: List[str]
    review_notes: Optional[str] = None


class ApplicationStatusResponse(BaseModel):
    """Response with application processing status."""
    application_id: str
    application_ref: str
    status: str
    processing_status: str
    current_agent: Optional[str] = None
    
    # Stage completion
    extraction_complete: bool = False
    validation_complete: bool = False
    eligibility_complete: bool = False
    recommendation_complete: bool = False
    
    # Counts
    documents_uploaded: int = 0
    validation_issues_count: int = 0
    
    # Error
    error: Optional[str] = None


class ApplicationResultResponse(BaseModel):
    """Response with complete application result."""
    application_id: str
    application_ref: str
    status: str
    
    # Documents
    uploaded_documents: List[DocumentInfo] = []
    
    # Extraction
    extraction_complete: bool
    extraction_success_count: int = 0
    
    # Validation
    validation_complete: bool
    validation_passed: bool
    validation_issues: List[ValidationIssue] = []
    
    # Eligibility
    eligibility_complete: bool
    eligibility_category: Optional[str] = None
    eligibility_score: Optional[float] = None
    feature_importances: Optional[Dict[str, float]] = None
    
    # Recommendation
    recommendation_complete: bool
    recommendation: Optional[Recommendation] = None
    enablement_programs: List[EnablementProgram] = []
    
    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Error
    error: Optional[str] = None


class ChatMessageResponse(BaseModel):
    """Response to a chat message."""
    message: str
    type: str = Field(default="assistant", description="Message type: assistant, system, error")
    application_ref: Optional[str] = None
    processing_status: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    databases: Dict[str, bool]
    ollama_available: bool


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    status_code: int
