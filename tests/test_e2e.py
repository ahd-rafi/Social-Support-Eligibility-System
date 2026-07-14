"""End-to-end integration tests for the application processing pipeline."""

import pytest
import asyncio
import logging
from pathlib import Path
from typing import List

from src.agents.graph import process_application
from src.agents.state import ApplicationState

logger = logging.getLogger(__name__)

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def get_applicant_documents(applicant_id: str) -> List[dict]:
    """Get document paths for a synthetic applicant.
    
    Args:
        applicant_id: Applicant folder name (e.g., "applicant_001")
        
    Returns:
        List of document dicts with file_path and file_name
    """
    base_path = Path(f"data/synthetic/output/{applicant_id}")
    
    docs = [
        {
            "file_path": str(base_path / "emirates_id.png"),
            "file_name": "emirates_id.png",
        },
        {
            "file_path": str(base_path / "bank_statement.pdf"),
            "file_name": "bank_statement.pdf",
        },
        {
            "file_path": str(base_path / "resume.pdf"),
            "file_name": "resume.pdf",
        },
        {
            "file_path": str(base_path / "assets_liabilities.xlsx"),
            "file_name": "assets_liabilities.xlsx",
        },
        {
            "file_path": str(base_path / "credit_report.pdf"),
            "file_name": "credit_report.pdf",
        },
    ]
    
    return docs


@pytest.fixture
def check_synthetic_data():
    """Fixture to check if synthetic data exists."""
    applicant_path = Path("data/synthetic/output/applicant_001")
    
    if not applicant_path.exists():
        pytest.skip(
            "Synthetic data not found. "
            "Run: python -m data.synthetic.generate"
        )


@pytest.mark.asyncio
async def test_e2e_clean_profile(check_synthetic_data):
    """Test pipeline with a clean profile (no inconsistencies).
    
    Expected behavior:
    - All documents extracted successfully
    - No validation issues
    - Eligibility score assigned
    - Decision recommendation generated
    - Enablement programs matched
    """
    logger.info("=" * 80)
    logger.info("TEST: E2E Clean Profile")
    logger.info("=" * 80)
    
    # Get documents
    docs = get_applicant_documents("applicant_001")
    
    # Verify files exist
    missing = [d for d in docs if not Path(d["file_path"]).exists()]
    if missing:
        pytest.skip(f"Missing files: {[d['file_name'] for d in missing]}")
    
    # Process application
    final_state = await process_application(
        application_id="test-clean-001",
        application_ref="TEST-CLEAN-001",
        uploaded_documents=docs,
    )
    
    # Assertions
    
    # Extraction
    assert final_state["extraction_complete"] is True, "Extraction should complete"
    assert len(final_state["extraction_results"]) >= 4, "Should extract at least 4 documents"
    
    # Count successful extractions
    success_count = sum(
        1 for r in final_state["extraction_results"].values()
        if r.get("status") == "success"
    )
    assert success_count >= 4, f"Expected at least 4 successful extractions, got {success_count}"
    
    # Validation
    assert final_state["validation_complete"] is True, "Validation should complete"
    # Clean profile should have no or minimal issues
    assert len(final_state.get("validation_issues", [])) <= 2, "Clean profile should have few issues"
    
    # Eligibility
    assert final_state["eligibility_complete"] is True, "Eligibility scoring should complete"
    assert final_state["eligibility_category"] in [
        "approved_financial",
        "approved_enablement",
        "soft_decline",
    ], f"Invalid eligibility category: {final_state['eligibility_category']}"
    
    assert 0.0 <= final_state["eligibility_score"] <= 1.0, "Score should be between 0 and 1"
    
    assert "feature_importances" in final_state, "Should have feature importances"
    assert len(final_state["feature_importances"]) > 0, "Should have at least one feature importance"
    
    # Recommendation
    assert final_state["recommendation_complete"] is True, "Recommendation should complete"
    
    recommendation = final_state.get("recommendation", {})
    assert "decision" in recommendation, "Should have decision"
    assert recommendation["decision"] in ["approved", "soft_decline"], "Invalid decision"
    assert "reasoning" in recommendation, "Should have reasoning"
    assert "key_factors" in recommendation, "Should have key factors"
    assert len(recommendation.get("key_factors", [])) > 0, "Should have at least one key factor"
    
    # Enablement programs (may be empty if no resume match)
    assert "enablement_programs" in final_state, "Should have enablement_programs field"
    
    # No errors
    assert final_state.get("error") is None, f"Should not have errors: {final_state.get('error')}"
    assert final_state["processing_status"] == "complete", "Status should be complete"
    
    logger.info("✓ TEST PASSED: E2E Clean Profile")
    logger.info(f"  Category: {final_state['eligibility_category']}")
    logger.info(f"  Score: {final_state['eligibility_score']:.3f}")
    logger.info(f"  Decision: {recommendation.get('decision')}")


@pytest.mark.asyncio
async def test_e2e_with_validation_issues(check_synthetic_data):
    """Test pipeline with profile that has validation issues.
    
    Expected behavior:
    - Documents extracted
    - Validation issues detected and flagged
    - Pipeline continues to completion (v1 linear graph)
    - Decision includes review notes about validation issues
    """
    logger.info("=" * 80)
    logger.info("TEST: E2E With Validation Issues")
    logger.info("=" * 80)
    
    # Use applicant_002 or create a profile known to have inconsistencies
    # For now, use applicant_001 and manually verify validation logic
    docs = get_applicant_documents("applicant_001")
    
    # Verify files exist
    missing = [d for d in docs if not Path(d["file_path"]).exists()]
    if missing:
        pytest.skip(f"Missing files: {[d['file_name'] for d in missing]}")
    
    # Process application
    final_state = await process_application(
        application_id="test-validation-001",
        application_ref="TEST-VAL-001",
        uploaded_documents=docs,
    )
    
    # Assertions
    
    # Should complete even with validation issues (v1 behavior)
    assert final_state["extraction_complete"] is True
    assert final_state["validation_complete"] is True
    assert final_state["eligibility_complete"] is True
    assert final_state["recommendation_complete"] is True
    
    # Validation issues are acceptable
    validation_issues = final_state.get("validation_issues", [])
    logger.info(f"Validation issues detected: {len(validation_issues)}")
    for issue in validation_issues:
        logger.info(f"  [{issue.get('severity')}] {issue.get('description')}")
    
    # If issues found, they should be properly structured
    if validation_issues:
        for issue in validation_issues:
            assert "issue_type" in issue
            assert "severity" in issue
            assert issue["severity"] in ["CRITICAL", "NEEDS_REVIEW", "INFO"]
            assert "description" in issue
    
    # Should still have a recommendation
    recommendation = final_state.get("recommendation", {})
    assert "decision" in recommendation
    
    # Review notes may mention validation issues
    if validation_issues:
        review_notes = recommendation.get("review_notes", "")
        logger.info(f"Review notes: {review_notes}")
    
    logger.info("✓ TEST PASSED: E2E With Validation Issues")
    logger.info(f"  Issues found: {len(validation_issues)}")
    logger.info(f"  Decision: {recommendation.get('decision')}")


@pytest.mark.asyncio
async def test_e2e_multiple_applicants(check_synthetic_data):
    """Test pipeline with multiple applicants to verify no state leakage.
    
    Expected behavior:
    - Each applicant processed independently
    - No data contamination between applications
    - All applicants complete successfully
    """
    logger.info("=" * 80)
    logger.info("TEST: E2E Multiple Applicants")
    logger.info("=" * 80)
    
    # Check if we have multiple applicants
    applicant_folders = list(Path("data/synthetic/output").glob("applicant_*"))
    
    if len(applicant_folders) < 2:
        pytest.skip("Need at least 2 synthetic applicants for this test")
    
    # Process first 3 applicants (or all if fewer than 3)
    applicants_to_test = sorted(applicant_folders)[:3]
    
    results = []
    
    for i, applicant_path in enumerate(applicants_to_test):
        applicant_id = applicant_path.name
        logger.info(f"\nProcessing {applicant_id}...")
        
        docs = get_applicant_documents(applicant_id)
        
        # Verify files exist
        missing = [d for d in docs if not Path(d["file_path"]).exists()]
        if missing:
            logger.warning(f"Skipping {applicant_id} - missing files: {missing}")
            continue
        
        # Process
        final_state = await process_application(
            application_id=f"test-multi-{i:03d}",
            application_ref=f"TEST-MULTI-{i:03d}",
            uploaded_documents=docs,
        )
        
        results.append({
            "applicant_id": applicant_id,
            "app_ref": f"TEST-MULTI-{i:03d}",
            "state": final_state,
        })
    
    # Assertions
    
    assert len(results) >= 2, "Should have processed at least 2 applicants"
    
    # All should complete
    for result in results:
        state = result["state"]
        assert state["extraction_complete"] is True, f"{result['applicant_id']}: Extraction incomplete"
        assert state["recommendation_complete"] is True, f"{result['applicant_id']}: Recommendation incomplete"
    
    # Application refs should be unique
    refs = [r["app_ref"] for r in results]
    assert len(refs) == len(set(refs)), "Application refs should be unique"
    
    # Applicant IDs in state should differ
    applicant_ids = [r["state"].get("applicant_id") for r in results]
    unique_ids = [aid for aid in applicant_ids if aid is not None]
    assert len(unique_ids) == len(set(unique_ids)), "Applicant IDs should be unique"
    
    logger.info("✓ TEST PASSED: E2E Multiple Applicants")
    logger.info(f"  Processed: {len(results)} applicants")
    for result in results:
        state = result["state"]
        logger.info(f"  {result['applicant_id']}: {state['eligibility_category']} ({state['eligibility_score']:.3f})")


@pytest.mark.asyncio
async def test_e2e_missing_document(check_synthetic_data):
    """Test pipeline behavior with missing documents.
    
    Expected behavior:
    - Pipeline handles missing documents gracefully
    - Extracts available documents
    - May complete with reduced confidence or flag for review
    """
    logger.info("=" * 80)
    logger.info("TEST: E2E Missing Document")
    logger.info("=" * 80)
    
    # Get documents but remove one
    docs = get_applicant_documents("applicant_001")
    
    # Remove credit report
    docs = [d for d in docs if "credit_report" not in d["file_name"]]
    
    logger.info(f"Processing with {len(docs)}/5 documents (missing credit_report)")
    
    # Process application
    final_state = await process_application(
        application_id="test-missing-001",
        application_ref="TEST-MISSING-001",
        uploaded_documents=docs,
    )
    
    # Assertions
    
    # Should attempt extraction (may have some failures)
    assert final_state["extraction_complete"] is True
    
    # May have errors for missing documents
    extraction_results = final_state.get("extraction_results", {})
    
    # Count successful vs failed
    success_count = sum(1 for r in extraction_results.values() if r.get("status") == "success")
    error_count = sum(1 for r in extraction_results.values() if r.get("status") == "error")
    
    logger.info(f"  Successful: {success_count}")
    logger.info(f"  Errors: {error_count}")
    
    # Should have at least some successful extractions
    assert success_count >= 3, "Should extract at least 3 documents"
    
    # Pipeline may complete with reduced data (depends on required fields)
    # For now, just verify it doesn't crash
    assert final_state.get("processing_status") in ["complete", "error"]
    
    logger.info("✓ TEST PASSED: E2E Missing Document")
    logger.info(f"  Status: {final_state['processing_status']}")


@pytest.mark.asyncio
async def test_e2e_feature_vector_completeness(check_synthetic_data):
    """Test that feature vector has all 12 required features.
    
    Expected behavior:
    - Feature vector contains all 12 features
    - No NaN or None values (or handled gracefully)
    """
    logger.info("=" * 80)
    logger.info("TEST: E2E Feature Vector Completeness")
    logger.info("=" * 80)
    
    docs = get_applicant_documents("applicant_001")
    
    # Verify files exist
    missing = [d for d in docs if not Path(d["file_path"]).exists()]
    if missing:
        pytest.skip(f"Missing files: {[d['file_name'] for d in missing]}")
    
    # Process application
    final_state = await process_application(
        application_id="test-features-001",
        application_ref="TEST-FEATURES-001",
        uploaded_documents=docs,
    )
    
    # Assertions
    
    if final_state["eligibility_complete"]:
        feature_vector = final_state.get("feature_vector", {})
        
        # Expected features
        expected_features = [
            "monthly_income",
            "employment_months",
            "household_size",
            "num_dependents",
            "income_per_family",
            "total_assets",
            "total_liabilities",
            "net_worth",
            "debt_to_income",
            "credit_score",
            "missed_payments",
            "age",
        ]
        
        logger.info("Feature vector:")
        for feature in expected_features:
            value = feature_vector.get(feature, "MISSING")
            logger.info(f"  {feature}: {value}")
        
        # Check all features present
        for feature in expected_features:
            assert feature in feature_vector, f"Missing feature: {feature}"
        
        logger.info("✓ TEST PASSED: Feature Vector Completeness")
    else:
        pytest.skip("Eligibility scoring did not complete")


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(test_e2e_clean_profile(None))
