"""LangGraph application processing pipeline."""

import logging
from typing import Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, END

from src.agents.state import ApplicationState, create_initial_state
from src.agents.extraction import extract_data_agent
from src.agents.validation import validate_data_agent
from src.agents.eligibility import eligibility_check_agent
from src.agents.recommendation import decision_recommendation_agent
from src.observability.tracing import get_tracing_client

logger = logging.getLogger(__name__)


def create_application_graph() -> StateGraph:
    """Create the LangGraph application processing pipeline.
    
    v1 graph: Strictly linear
    START → extraction → validation → eligibility → decision → END
    
    No conditional routing. Validation flags issues but always proceeds forward.
    This is a deliberate scope decision for the Tuesday deadline.
    
    Future v2 improvement: Add conditional edge from validation back to extraction
    with bounded retry (max 1 attempt) for extraction errors.
    
    Returns:
        Compiled StateGraph
    """
    logger.info("Creating application processing graph (v1 - linear)")
    
    # Create graph with ApplicationState schema
    graph = StateGraph(ApplicationState)
    
    # Add nodes
    graph.add_node("extraction", extract_data_agent)
    graph.add_node("validation", validate_data_agent)
    graph.add_node("eligibility", eligibility_check_agent)
    graph.add_node("decision", decision_recommendation_agent)  # Renamed to avoid state key conflict
    
    # Add edges (v1: linear pipeline)
    graph.set_entry_point("extraction")
    graph.add_edge("extraction", "validation")
    graph.add_edge("validation", "eligibility")
    graph.add_edge("eligibility", "decision")
    graph.add_edge("decision", END)
    
    logger.info("Graph structure:")
    logger.info("  START → extraction → validation → eligibility → decision → END")
    logger.info("  (v1: No retry loop, linear progression)")
    
    # Compile graph
    return graph.compile()


# Global instance
_app_graph = None


def get_application_graph() -> StateGraph:
    """Get or create global graph instance.
    
    Returns:
        Compiled StateGraph
    """
    global _app_graph
    if _app_graph is None:
        _app_graph = create_application_graph()
        logger.info("Application graph compiled and cached")
    return _app_graph


async def process_application(
    application_id: str,
    application_ref: str,
    uploaded_documents: list,
) -> ApplicationState:
    """Process an application through the full pipeline.
    
    This is the main entry point for running the agent graph.
    
    Args:
        application_id: Unique application ID (UUID)
        application_ref: Human-readable reference (e.g., APP-001)
        uploaded_documents: List of uploaded document dicts with file_path, file_name
        
    Returns:
        Final ApplicationState with complete recommendation
    """
    logger.info("=" * 100)
    logger.info(f"STARTING APPLICATION PROCESSING: {application_ref}")
    logger.info("=" * 100)
    
    # Create pipeline trace
    tracing_client = get_tracing_client()
    pipeline_trace_id = tracing_client.create_trace(
        name="application_pipeline",
        session_id=application_id,
        metadata={
            "application_ref": application_ref,
            "document_count": len(uploaded_documents),
            "pipeline_version": "v1_linear",
        },
    )
    
    # Initialize state
    initial_state = create_initial_state(application_id, application_ref)
    initial_state["uploaded_documents"] = uploaded_documents
    initial_state["started_at"] = datetime.utcnow().isoformat()
    
    logger.info(f"Application ID: {application_id}")
    logger.info(f"Application Ref: {application_ref}")
    logger.info(f"Documents: {len(uploaded_documents)}")
    if pipeline_trace_id:
        logger.info(f"Trace ID: {pipeline_trace_id}")
    
    # Get compiled graph
    graph = get_application_graph()
    
    try:
        # Run graph
        logger.info("")
        logger.info("Running agent pipeline...")
        logger.info("")
        
        final_state = await graph.ainvoke(initial_state)
        
        # Check for errors
        if final_state.get("error"):
            logger.error(f"Pipeline completed with error: {final_state['error']}")
            
            # Log error event to trace
            if pipeline_trace_id:
                tracing_client.log_event(
                    trace_id=pipeline_trace_id,
                    name="pipeline_error",
                    level="ERROR",
                    metadata={"error": final_state.get("error")},
                )
        else:
            logger.info("")
            logger.info("=" * 100)
            logger.info("✓ PIPELINE COMPLETE - SUMMARY")
            logger.info("=" * 100)
            logger.info(f"  Application: {application_ref}")
            logger.info(f"  Extraction: {'✓' if final_state.get('extraction_complete') else '✗'}")
            logger.info(f"  Validation: {'✓' if final_state.get('validation_complete') else '✗'} ({len(final_state.get('validation_issues', []))} issues)")
            logger.info(f"  Eligibility: {final_state.get('eligibility_category', 'N/A')} ({final_state.get('eligibility_score', 0):.3f})")
            logger.info(f"  Decision: {final_state.get('recommendation', {}).get('decision', 'N/A')}")
            logger.info(f"  Programs: {len(final_state.get('enablement_programs', []))}")
            logger.info("=" * 100)
            
            # Log completion event to trace
            if pipeline_trace_id:
                tracing_client.log_event(
                    trace_id=pipeline_trace_id,
                    name="pipeline_complete",
                    level="DEFAULT",
                    metadata={
                        "decision": final_state.get("recommendation", {}).get("decision"),
                        "eligibility_category": final_state.get("eligibility_category"),
                        "eligibility_score": final_state.get("eligibility_score"),
                        "validation_issues_count": len(final_state.get("validation_issues", [])),
                        "programs_matched": len(final_state.get("enablement_programs", [])),
                    },
                )
        
        # Flush traces
        tracing_client.flush()
        
        return final_state
        
    except Exception as e:
        logger.error(f"Pipeline failed with exception: {e}", exc_info=True)
        
        # Log error to trace
        if pipeline_trace_id:
            tracing_client.log_event(
                trace_id=pipeline_trace_id,
                name="pipeline_exception",
                level="ERROR",
                metadata={"exception": str(e)},
            )
            tracing_client.flush()
        
        # Return state with error
        error_state = initial_state.copy()
        error_state["error"] = str(e)
        error_state["processing_status"] = "error"
        return error_state


# Utility Functions

def format_recommendation_for_chat(state: ApplicationState) -> str:
    """Format the final recommendation for chat UI display.
    
    Args:
        state: Final application state
        
    Returns:
        Formatted recommendation text
    """
    recommendation = state.get("recommendation", {})
    enablement_programs = state.get("enablement_programs", [])
    validation_issues = state.get("validation_issues", [])
    
    # Build formatted output
    lines = []
    lines.append("# Assessment Complete ✓")
    lines.append("")
    
    # Decision
    decision = recommendation.get("decision", "unknown")
    if decision == "approved":
        lines.append("**Your application has been recommended for approval.**")
    elif decision == "soft_decline":
        lines.append("**Your application does not currently qualify.**")
    else:
        lines.append(f"**Decision: {decision}**")
    
    lines.append("")
    
    # Reasoning
    reasoning = recommendation.get("reasoning", "")
    if reasoning:
        lines.append("## Reasoning")
        lines.append(reasoning)
        lines.append("")
    
    # Key factors
    key_factors = recommendation.get("key_factors", [])
    if key_factors:
        lines.append("## Key Factors")
        for factor in key_factors:
            lines.append(f"- {factor}")
        lines.append("")
    
    # Enablement programs
    if enablement_programs:
        lines.append("## Recommended Programs")
        lines.append("Based on your profile, we recommend the following programs:")
        lines.append("")
        for i, prog in enumerate(enablement_programs[:5], 1):
            match_score = prog.get("match_score", 0)
            lines.append(f"**{i}. {prog.get('program_name')}** ({match_score*100:.0f}% match)")
            if prog.get("description"):
                lines.append(f"   {prog.get('description')}")
            if prog.get("provider"):
                lines.append(f"   *Provider: {prog.get('provider')}*")
            lines.append("")
    
    # Validation issues (if any)
    if validation_issues:
        lines.append("## Items for Review")
        lines.append("The following items were flagged for case worker review:")
        lines.append("")
        for issue in validation_issues:
            severity = issue.get("severity", "INFO")
            description = issue.get("description", "")
            lines.append(f"- **[{severity}]** {description}")
        lines.append("")
    
    # Review notes
    review_notes = recommendation.get("review_notes", "")
    if review_notes:
        lines.append("## Case Worker Notes")
        lines.append(review_notes)
        lines.append("")
    
    # Reference
    lines.append("---")
    lines.append(f"**Reference:** {state.get('application_ref', 'N/A')}")
    lines.append(f"**Confidence:** {recommendation.get('confidence', 0):.0%}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    import asyncio
    from pathlib import Path
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    async def test():
        """Test the complete pipeline with sample documents."""
        
        # Sample documents (update paths as needed)
        docs = [
            {
                "file_path": "data/synthetic/output/applicant_001/emirates_id.png",
                "file_name": "emirates_id.png",
            },
            {
                "file_path": "data/synthetic/output/applicant_001/bank_statement.pdf",
                "file_name": "bank_statement.pdf",
            },
            {
                "file_path": "data/synthetic/output/applicant_001/resume.pdf",
                "file_name": "resume.pdf",
            },
            {
                "file_path": "data/synthetic/output/applicant_001/assets_liabilities.xlsx",
                "file_name": "assets_liabilities.xlsx",
            },
            {
                "file_path": "data/synthetic/output/applicant_001/credit_report.pdf",
                "file_name": "credit_report.pdf",
            },
        ]
        
        # Check if files exist
        missing = [d for d in docs if not Path(d["file_path"]).exists()]
        if missing:
            print(f"Missing files: {[d['file_name'] for d in missing]}")
            print("Please generate synthetic data first: python -m data.synthetic.generate")
            return
        
        # Process application
        final_state = await process_application(
            application_id="test-app-001",
            application_ref="APP-TEST-001",
            uploaded_documents=docs,
        )
        
        # Format and print result
        print("\n")
        print("=" * 100)
        print("FORMATTED RECOMMENDATION FOR CHAT UI:")
        print("=" * 100)
        print(format_recommendation_for_chat(final_state))
    
    asyncio.run(test())
