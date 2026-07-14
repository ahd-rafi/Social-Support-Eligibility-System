"""Streamlit chat UI for social support application."""

import streamlit as st
import requests
import time
from pathlib import Path
from typing import Dict, Any, List

# API configuration
API_BASE_URL = "http://localhost:8000/api"

# Page configuration
st.set_page_config(
    page_title="Social Support Application",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FFFFFF;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #E5E7EB;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-pending {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
    }
    .status-processing {
        background-color: #DBEAFE;
        border-left: 4px solid #3B82F6;
    }
    .status-complete {
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
    }
    .status-error {
        background-color: #FEE2E2;
        border-left: 4px solid #EF4444;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if "application_id" not in st.session_state:
        st.session_state.application_id = None
    if "application_ref" not in st.session_state:
        st.session_state.application_ref = None
    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = []
    if "processing_started" not in st.session_state:
        st.session_state.processing_started = False
    if "result" not in st.session_state:
        st.session_state.result = None


def check_api_health() -> bool:
    """Check if API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def create_application(name: str = None, household_size: int = None) -> Dict[str, Any]:
    """Create a new application."""
    payload = {}
    if name:
        payload["applicant_name"] = name
    if household_size:
        payload["household_size"] = household_size
    
    response = requests.post(f"{API_BASE_URL}/applications", json=payload)
    response.raise_for_status()
    return response.json()


def upload_document(application_id: str, file, document_type: str) -> Dict[str, Any]:
    """Upload a document."""
    files = {"file": (file.name, file.getvalue(), file.type)}
    data = {"document_type": document_type}
    
    response = requests.post(
        f"{API_BASE_URL}/applications/{application_id}/upload",
        files=files,
        data=data,
    )
    response.raise_for_status()
    return response.json()


def start_processing(application_id: str) -> Dict[str, Any]:
    """Start application processing."""
    response = requests.post(f"{API_BASE_URL}/applications/{application_id}/process")
    response.raise_for_status()
    return response.json()


def get_status(application_id: str) -> Dict[str, Any]:
    """Get application status."""
    response = requests.get(f"{API_BASE_URL}/applications/{application_id}/status")
    response.raise_for_status()
    return response.json()


def get_result(application_id: str) -> Dict[str, Any]:
    """Get application result."""
    response = requests.get(f"{API_BASE_URL}/applications/{application_id}/result")
    response.raise_for_status()
    return response.json()


def render_header():
    """Render application header."""
    st.markdown('<div class="main-header">🏛️ Social Support Application System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered eligibility assessment for government social support programs</div>', unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with info."""
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This system processes your application for social support using AI to:
        
        - Extract information from your documents
        - Validate data consistency
        - Assess eligibility
        - Recommend programs
        
        **Required Documents:**
        - Emirates ID
        - Bank Statement (last 3-6 months)
        - Resume/CV
        - Assets & Liabilities (Excel)
        - Credit Report
        """)
        
        st.divider()
        
        # API status
        api_healthy = check_api_health()
        if api_healthy:
            st.success("✓ API Connected")
        else:
            st.error("✗ API Unavailable")
        
        if st.session_state.application_ref:
            st.divider()
            st.info(f"**Application:** {st.session_state.application_ref}")


def render_upload_stage():
    """Render document upload stage."""
    st.header("📤 Step 1: Upload Documents")
    
    # Create application if not exists
    if not st.session_state.application_id:
        st.write("Let's start your application. You can provide some basic information now, or we'll extract it from your documents.")
        
        with st.form("create_application"):
            name = st.text_input("Your Name", placeholder="e.g., Ahmed Al-Mansoori")
            household_size = st.number_input("Household Size", min_value=1, max_value=20, value=1, help="Total number of people in your household including yourself")
            
            submitted = st.form_submit_button("Create Application")
            
            if submitted:
                with st.spinner("Creating application..."):
                    try:
                        result = create_application(name if name else None, household_size)
                        st.session_state.application_id = result["application_id"]
                        st.session_state.application_ref = result["application_ref"]
                        st.success(f"✓ Application created: {result['application_ref']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to create application: {e}")
        
        return
    
    # Document upload
    st.write(f"**Application:** {st.session_state.application_ref}")
    
    document_types = {
        "Emirates ID": "emirates_id",
        "Bank Statement": "bank_statement",
        "Resume/CV": "resume",
        "Assets & Liabilities": "assets_liabilities",
        "Credit Report": "credit_report",
    }
    
    uploaded_types = [doc["type"] for doc in st.session_state.uploaded_documents]
    
    # Show progress
    progress = len(uploaded_types) / len(document_types)
    st.progress(progress, text=f"Uploaded {len(uploaded_types)}/{len(document_types)} documents")
    
    # Upload form
    for doc_label, doc_type in document_types.items():
        if doc_type in uploaded_types:
            st.success(f"✓ {doc_label} uploaded")
        else:
            uploaded_file = st.file_uploader(
                f"Upload {doc_label}",
                key=f"upload_{doc_type}",
                help=f"Upload your {doc_label}",
            )
            
            if uploaded_file:
                with st.spinner(f"Uploading {doc_label}..."):
                    try:
                        result = upload_document(
                            st.session_state.application_id,
                            uploaded_file,
                            doc_type,
                        )
                        st.session_state.uploaded_documents.append({
                            "name": uploaded_file.name,
                            "type": doc_type,
                        })
                        st.success(result["message"])
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to upload: {e}")
    
    # Start processing button
    if len(uploaded_types) == len(document_types):
        st.divider()
        st.success("✓ All documents uploaded!")
        
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            with st.spinner("Starting processing..."):
                try:
                    result = start_processing(st.session_state.application_id)
                    st.session_state.processing_started = True
                    st.success("Processing started!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start processing: {e}")


def render_processing_stage():
    """Render processing status stage."""
    st.header("⚙️ Step 2: Processing Application")
    
    # Get status
    try:
        status = get_status(st.session_state.application_id)
    except Exception as e:
        st.error(f"Failed to get status: {e}")
        return
    
    # Progress indicators
    stages = [
        ("Extraction", status["extraction_complete"]),
        ("Validation", status["validation_complete"]),
        ("Eligibility", status["eligibility_complete"]),
        ("Recommendation", status["recommendation_complete"]),
    ]
    
    cols = st.columns(4)
    for col, (stage_name, complete) in zip(cols, stages):
        with col:
            if complete:
                st.success(f"✓ {stage_name}")
            elif status["current_agent"] and stage_name.lower() in status["current_agent"]:
                st.info(f"⏳ {stage_name}")
            else:
                st.text(f"⚪ {stage_name}")
    
    # Status details
    st.divider()
    
    if status["processing_status"] == "complete":
        st.success("✓ Processing complete!")
        if st.button("View Results", type="primary", use_container_width=True):
            try:
                result = get_result(st.session_state.application_id)
                st.session_state.result = result
                st.rerun()
            except Exception as e:
                st.error(f"Failed to get results: {e}")
    
    elif status["processing_status"] == "error":
        st.error(f"✗ Processing failed: {status['error']}")
    
    else:
        st.info(f"Processing... Current stage: {status['current_agent'] or 'initializing'}")
        
        # Auto-refresh
        time.sleep(2)
        st.rerun()


def render_result_stage():
    """Render final result stage."""
    st.header("📊 Step 3: Assessment Results")
    
    result = st.session_state.result
    
    # Decision
    recommendation = result.get("recommendation")
    if recommendation:
        decision = recommendation["decision"]
        confidence = recommendation["confidence"]
        
        if decision == "approved":
            st.success(f"✅ **Application Approved** (Confidence: {confidence:.0%})")
        else:
            st.warning(f"⚠️ **Application Under Review** (Confidence: {confidence:.0%})")
        
        # Reasoning
        st.write("### Reasoning")
        st.write(recommendation["reasoning"])
        
        # Key factors
        if recommendation.get("key_factors"):
            st.write("### Key Factors")
            for factor in recommendation["key_factors"]:
                st.write(f"- {factor}")
    
    # Eligibility details
    st.divider()
    st.write("### Eligibility Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Category", result.get("eligibility_category", "N/A"))
    
    with col2:
        score = result.get("eligibility_score", 0)
        st.metric("Score", f"{score:.1%}")
    
    with col3:
        issues = len(result.get("validation_issues", []))
        st.metric("Validation Issues", issues)
    
    # Enablement programs
    programs = result.get("enablement_programs", [])
    if programs:
        st.divider()
        st.write("### 🎓 Recommended Programs")
        st.write("Based on your profile, we recommend these programs:")
        
        for prog in programs:
            with st.expander(f"**{prog['program_name']}** ({prog['match_score']:.0%} match)"):
                st.write(prog["description"])
                if prog.get("provider"):
                    st.write(f"**Provider:** {prog['provider']}")
                if prog.get("duration"):
                    st.write(f"**Duration:** {prog['duration']}")
    
    # Validation issues
    issues = result.get("validation_issues", [])
    if issues:
        st.divider()
        st.write("### ⚠️ Items for Review")
        
        for issue in issues:
            severity_color = {
                "CRITICAL": "🔴",
                "NEEDS_REVIEW": "🟡",
                "INFO": "🔵",
            }.get(issue["severity"], "⚪")
            
            st.write(f"{severity_color} **[{issue['severity']}]** {issue['description']}")
    
    # Review notes
    if recommendation and recommendation.get("review_notes"):
        st.divider()
        st.info(f"**Case Worker Notes:** {recommendation['review_notes']}")
    
    # Actions
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Report", use_container_width=True):
            st.info("Report download feature coming soon!")
    
    with col2:
        if st.button("🔄 New Application", use_container_width=True):
            # Reset session
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def main():
    """Main application."""
    initialize_session_state()
    
    render_header()
    render_sidebar()
    
    st.divider()
    
    # Determine stage
    if st.session_state.result:
        render_result_stage()
    elif st.session_state.processing_started:
        render_processing_stage()
    else:
        render_upload_stage()


if __name__ == "__main__":
    main()
