"""API routes for application processing."""

import logging
import uuid
from pathlib import Path
from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse

from src.config import settings
from src.api.schemas import (
    CreateApplicationRequest,
    ApplicationResponse,
    UploadDocumentResponse,
    ApplicationStatusResponse,
    ApplicationResultResponse,
    ChatMessageResponse,
    HealthCheckResponse,
    DocumentInfo,
    ValidationIssue,
    EnablementProgram,
    Recommendation,
)
from src.agents.graph import process_application
from src.agents.llm import get_llm

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for application states (would be Redis/PostgreSQL in production)
application_states = {}


# Health Check

@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Check API and database health."""
    from src.api.main import postgres_client, mongo_client, neo4j_client, qdrant_client
    
    databases = {
        "postgresql": False,
        "mongodb": False,
        "neo4j": False,
        "qdrant": False,
    }
    
    # Check PostgreSQL
    try:
        await postgres_client.fetchval("SELECT 1")
        databases["postgresql"] = True
    except Exception:
        pass
    
    # Check MongoDB
    try:
        mongo_client.db.command("ping")
        databases["mongodb"] = True
    except Exception:
        pass
    
    # Check Neo4j
    try:
        neo4j_client.execute_query("RETURN 1")
        databases["neo4j"] = True
    except Exception:
        pass
    
    # Check Qdrant
    try:
        qdrant_client.client.get_collections()
        databases["qdrant"] = True
    except Exception:
        pass
    
    # Check Ollama
    ollama_available = False
    try:
        llm = get_llm()
        ollama_available = llm.is_available()
    except Exception:
        pass
    
    status = "healthy" if all(databases.values()) and ollama_available else "degraded"
    
    return HealthCheckResponse(
        status=status,
        version="1.0.0",
        databases=databases,
        ollama_available=ollama_available,
    )


# Application Management

@router.post("/applications", response_model=ApplicationResponse)
async def create_application(request: CreateApplicationRequest):
    """Create a new application.
    
    This initializes an application record and returns an application ID
    that can be used for document uploads.
    """
    from src.api.main import postgres_client
    
    try:
        # Generate IDs
        application_id = str(uuid.uuid4())
        application_ref = f"APP-{application_id[:8].upper()}"
        
        # Create application record in PostgreSQL
        # First, create or get applicant record
        applicant_id = str(uuid.uuid4())
        
        if request.applicant_name:
            await postgres_client.execute(
                """
                INSERT INTO applicants (id, name, emirates_id, dob, nationality, gender)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                applicant_id,
                request.applicant_name,
                "PENDING",  # Will be updated from Emirates ID
                None,  # Will be updated from Emirates ID
                "Unknown",
                None,
            )
        
        # Create application
        await postgres_client.execute(
            """
            INSERT INTO applications 
            (id, applicant_id, application_ref, status, household_size, num_dependents, monthly_income)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            application_id,
            applicant_id if request.applicant_name else None,
            application_ref,
            "pending",
            request.household_size,
            request.num_dependents,
            request.monthly_income,
        )
        
        # Initialize in-memory state
        application_states[application_id] = {
            "application_id": application_id,
            "application_ref": application_ref,
            "status": "pending",
            "uploaded_documents": [],
            "processing_status": "pending",
            "created_at": datetime.utcnow(),
        }
        
        logger.info(f"Created application: {application_ref}")
        
        return ApplicationResponse(
            application_id=application_id,
            application_ref=application_ref,
            status="pending",
            created_at=datetime.utcnow(),
        )
        
    except Exception as e:
        logger.error(f"Failed to create application: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/applications/{application_id}/upload", response_model=UploadDocumentResponse)
async def upload_document(
    application_id: str,
    file: UploadFile = File(...),
    document_type: str = Form(...),
):
    """Upload a document for an application.
    
    Supported document types:
    - emirates_id (image: PNG, JPG)
    - bank_statement (PDF)
    - resume (PDF, DOCX)
    - credit_report (PDF)
    - assets_liabilities (XLSX)
    """
    try:
        # Verify application exists
        if application_id not in application_states:
            raise HTTPException(status_code=404, detail="Application not found")
        
        # Create upload directory
        upload_dir = settings.upload_dir / application_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Update application state
        application_states[application_id]["uploaded_documents"].append({
            "file_path": str(file_path),
            "file_name": file.filename,
            "document_type": document_type,
            "uploaded_at": datetime.utcnow(),
        })
        
        document_count = len(application_states[application_id]["uploaded_documents"])
        
        logger.info(f"Uploaded document {file.filename} for application {application_id}")
        
        return UploadDocumentResponse(
            success=True,
            message=f"Document {file.filename} uploaded successfully",
            document_count=document_count,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/applications/{application_id}/process")
async def process_application_endpoint(
    application_id: str,
    background_tasks: BackgroundTasks,
):
    """Start processing an application through the agent pipeline.
    
    This runs the full LangGraph pipeline in the background:
    extraction → validation → eligibility → recommendation
    """
    try:
        # Verify application exists
        if application_id not in application_states:
            raise HTTPException(status_code=404, detail="Application not found")
        
        state = application_states[application_id]
        
        # Verify documents uploaded
        if not state.get("uploaded_documents"):
            raise HTTPException(status_code=400, detail="No documents uploaded")
        
        # Check if already processing
        if state.get("processing_status") not in ["pending", "error"]:
            raise HTTPException(status_code=400, detail=f"Application already {state['processing_status']}")
        
        # Update status
        state["processing_status"] = "processing"
        
        # Process in background
        async def run_pipeline():
            try:
                logger.info(f"Starting pipeline for application {application_id}")
                
                final_state = await process_application(
                    application_id=application_id,
                    application_ref=state["application_ref"],
                    uploaded_documents=state["uploaded_documents"],
                )
                
                # Update stored state
                application_states[application_id].update({
                    "processing_status": final_state.get("processing_status", "complete"),
                    "extraction_complete": final_state.get("extraction_complete", False),
                    "validation_complete": final_state.get("validation_complete", False),
                    "eligibility_complete": final_state.get("eligibility_complete", False),
                    "recommendation_complete": final_state.get("recommendation_complete", False),
                    "final_state": final_state,
                    "completed_at": datetime.utcnow(),
                })
                
                logger.info(f"Pipeline complete for application {application_id}")
                
            except Exception as e:
                logger.error(f"Pipeline failed for application {application_id}: {e}", exc_info=True)
                application_states[application_id]["processing_status"] = "error"
                application_states[application_id]["error"] = str(e)
        
        background_tasks.add_task(run_pipeline)
        
        return JSONResponse(
            content={
                "message": "Processing started",
                "application_id": application_id,
                "application_ref": state["application_ref"],
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/applications/{application_id}/status", response_model=ApplicationStatusResponse)
async def get_application_status(application_id: str):
    """Get current processing status of an application."""
    try:
        if application_id not in application_states:
            raise HTTPException(status_code=404, detail="Application not found")
        
        state = application_states[application_id]
        final_state = state.get("final_state", {})
        
        return ApplicationStatusResponse(
            application_id=application_id,
            application_ref=state["application_ref"],
            status=state["status"],
            processing_status=state.get("processing_status", "pending"),
            current_agent=final_state.get("current_agent"),
            extraction_complete=state.get("extraction_complete", False),
            validation_complete=state.get("validation_complete", False),
            eligibility_complete=state.get("eligibility_complete", False),
            recommendation_complete=state.get("recommendation_complete", False),
            documents_uploaded=len(state.get("uploaded_documents", [])),
            validation_issues_count=len(final_state.get("validation_issues", [])),
            error=state.get("error"),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/applications/{application_id}/result", response_model=ApplicationResultResponse)
async def get_application_result(application_id: str):
    """Get complete result of a processed application."""
    try:
        if application_id not in application_states:
            raise HTTPException(status_code=404, detail="Application not found")
        
        state = application_states[application_id]
        final_state = state.get("final_state", {})
        
        # Check if processing complete
        if state.get("processing_status") not in ["complete", "error"]:
            raise HTTPException(status_code=400, detail="Application still processing")
        
        # Format uploaded documents
        uploaded_docs = [
            DocumentInfo(
                file_name=doc["file_name"],
                file_type=doc.get("document_type", "unknown"),
                document_type=doc.get("document_type", "unknown"),
                uploaded_at=doc.get("uploaded_at", datetime.utcnow()),
            )
            for doc in state.get("uploaded_documents", [])
        ]
        
        # Format validation issues
        validation_issues = [
            ValidationIssue(**issue)
            for issue in final_state.get("validation_issues", [])
        ]
        
        # Format recommendation
        recommendation = None
        if final_state.get("recommendation"):
            rec = final_state["recommendation"]
            recommendation = Recommendation(
                decision=rec.get("decision", "unknown"),
                confidence=rec.get("confidence", 0.0),
                reasoning=rec.get("reasoning", ""),
                key_factors=rec.get("key_factors", []),
                review_notes=rec.get("review_notes"),
            )
        
        # Format enablement programs
        enablement_programs = [
            EnablementProgram(**prog)
            for prog in final_state.get("enablement_programs", [])
        ]
        
        # Count extraction successes
        extraction_results = final_state.get("extraction_results", {})
        extraction_success_count = sum(
            1 for r in extraction_results.values()
            if r.get("status") == "success"
        )
        
        return ApplicationResultResponse(
            application_id=application_id,
            application_ref=state["application_ref"],
            status=state["status"],
            uploaded_documents=uploaded_docs,
            extraction_complete=final_state.get("extraction_complete", False),
            extraction_success_count=extraction_success_count,
            validation_complete=final_state.get("validation_complete", False),
            validation_passed=final_state.get("validation_passed", False),
            validation_issues=validation_issues,
            eligibility_complete=final_state.get("eligibility_complete", False),
            eligibility_category=final_state.get("eligibility_category"),
            eligibility_score=final_state.get("eligibility_score"),
            feature_importances=final_state.get("feature_importances"),
            recommendation_complete=final_state.get("recommendation_complete", False),
            recommendation=recommendation,
            enablement_programs=enablement_programs,
            started_at=state.get("created_at"),
            completed_at=state.get("completed_at"),
            error=final_state.get("error"),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get result: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Chat Interface

@router.post("/chat", response_model=ChatMessageResponse)
async def chat_message(application_id: str, message: str):
    """Send a chat message (for future chat-based interaction).
    
    Currently returns status updates. In production, would handle
    conversational document collection.
    """
    try:
        if application_id not in application_states:
            return ChatMessageResponse(
                message="I couldn't find that application. Please create a new application first.",
                type="error",
            )
        
        state = application_states[application_id]
        processing_status = state.get("processing_status", "pending")
        
        # Simple status-based responses
        if processing_status == "pending":
            response = "Your application is ready. Please upload your documents to begin processing."
        elif processing_status == "processing":
            current_agent = state.get("final_state", {}).get("current_agent", "unknown")
            response = f"Your application is being processed. Current stage: {current_agent}"
        elif processing_status == "complete":
            final_state = state.get("final_state", {})
            decision = final_state.get("recommendation", {}).get("decision", "unknown")
            response = f"Your application has been processed. Decision: {decision}. You can view the full result at /api/applications/{application_id}/result"
        elif processing_status == "error":
            error = state.get("error", "Unknown error")
            response = f"An error occurred while processing your application: {error}"
        else:
            response = f"Application status: {processing_status}"
        
        return ChatMessageResponse(
            message=response,
            type="assistant",
            application_ref=state["application_ref"],
            processing_status=processing_status,
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return ChatMessageResponse(
            message="I encountered an error. Please try again.",
            type="error",
        )
