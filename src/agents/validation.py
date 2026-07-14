"""Data Validation Agent - Reflexion pattern implementation."""

import logging
from typing import Dict, Any, List, Optional

from src.agents.state import ApplicationState
from src.agents.llm import get_llm
from src.db.postgres import PostgresClient
from src.db.neo4j_client import Neo4jClient
from src.observability.tracing import trace_agent_start, trace_agent_end, trace_validation_issue

logger = logging.getLogger(__name__)


class ValidationAgent:
    """Data Validation Agent using Reflexion pattern.
    
    Cross-document consistency validation:
    1. Initial validation checks (address, income, employment, family)
    2. Reflexion self-critique: "Did I miss anything?"
    3. Revised validation with additional checks
    4. Log all issues with severity levels
    
    v1: Linear graph - always proceeds forward even if issues found.
    Issues are flagged and logged for human review.
    """

    def __init__(self):
        """Initialize validation agent."""
        self.llm = get_llm()
        self.postgres = PostgresClient()
        self.neo4j = Neo4jClient()
        
        logger.info("Validation agent initialized")

    async def connect_databases(self):
        """Connect to databases."""
        await self.postgres.connect()
        self.neo4j.connect()

    async def disconnect_databases(self):
        """Disconnect from databases."""
        await self.postgres.disconnect()
        self.neo4j.disconnect()

    async def check_address_consistency(
        self,
        applicant_id: str,
    ) -> List[Dict[str, Any]]:
        """Check address consistency across Emirates ID and credit report.
        
        Args:
            applicant_id: Applicant ID
            
        Returns:
            List of issues found
        """
        logger.debug("[Validation] Checking address consistency")
        
        query = """
        MATCH (p:Person {id: $applicant_id})-[:LIVES_AT]->(addr:Address)<-[:source]-(source)
        RETURN addr.emirate as emirate, source as document_source
        """
        
        try:
            results = self.neo4j.execute_query(
                """
                MATCH (p:Person {id: $applicant_id})-[r:LIVES_AT]->(addr:Address)
                RETURN addr.emirate as emirate, r.source as source, addr.street as street
                """,
                {"applicant_id": applicant_id},
            )
            
            if len(results) < 2:
                # Not enough addresses to compare
                return []
            
            # Group by source
            addresses_by_source = {}
            for row in results:
                source = row.get("source", "unknown")
                addresses_by_source[source] = {
                    "emirate": row.get("emirate"),
                    "street": row.get("street"),
                }
            
            # Compare emirates_id vs credit_report addresses
            if "emirates_id" in addresses_by_source and "credit_report" in addresses_by_source:
                id_emirate = addresses_by_source["emirates_id"]["emirate"]
                cr_emirate = addresses_by_source["credit_report"]["emirate"]
                
                if id_emirate != cr_emirate:
                    return [{
                        "issue_type": "address_mismatch",
                        "severity": "CRITICAL",
                        "field_name": "emirate",
                        "source_doc_a": "emirates_id",
                        "source_doc_b": "credit_report",
                        "value_a": id_emirate,
                        "value_b": cr_emirate,
                        "description": f"Emirate mismatch: Emirates ID shows '{id_emirate}' but credit report shows '{cr_emirate}'",
                    }]
            
            return []
            
        except Exception as e:
            logger.error(f"Address consistency check failed: {e}")
            return []

    async def check_income_consistency(
        self,
        application_id: str,
    ) -> List[Dict[str, Any]]:
        """Check income consistency across form and bank statement.
        
        Args:
            application_id: Application ID
            
        Returns:
            List of issues found
        """
        logger.debug("[Validation] Checking income consistency")
        
        try:
            # Get reported income from application form
            app_result = await self.postgres.fetchone(
                """
                SELECT monthly_income
                FROM applications
                WHERE id = $1
                """,
                application_id,
            )
            
            if not app_result:
                return []
            
            reported_income = float(app_result.get("monthly_income", 0) or 0)
            
            # Get bank statement income from extraction
            bank_result = await self.postgres.fetchone(
                """
                SELECT extraction_data->'total_credits' as bank_income
                FROM extraction_results
                WHERE application_id = $1 AND document_type = 'bank_statement'
                """,
                application_id,
            )
            
            if not bank_result or not bank_result.get("bank_income"):
                return []
            
            bank_income = float(bank_result.get("bank_income", 0))
            
            # Check for significant mismatch (>20% difference)
            if reported_income > 0 and bank_income > 0:
                diff_pct = abs(reported_income - bank_income) / reported_income
                
                if diff_pct > 0.20:  # More than 20% difference
                    return [{
                        "issue_type": "income_mismatch",
                        "severity": "CRITICAL",
                        "field_name": "monthly_income",
                        "source_doc_a": "application_form",
                        "source_doc_b": "bank_statement",
                        "value_a": str(reported_income),
                        "value_b": str(bank_income),
                        "description": f"Income mismatch: Form reports {reported_income:.2f} AED but bank statement shows {bank_income:.2f} AED ({diff_pct*100:.1f}% difference)",
                    }]
            
            return []
            
        except Exception as e:
            logger.error(f"Income consistency check failed: {e}")
            return []

    async def check_employment_consistency(
        self,
        applicant_id: str,
    ) -> List[Dict[str, Any]]:
        """Check employment consistency across resume and bank statement.
        
        Args:
            applicant_id: Applicant ID
            
        Returns:
            List of issues found
        """
        logger.debug("[Validation] Checking employment consistency")
        
        try:
            # Get employers from different sources
            results = self.neo4j.execute_query(
                """
                MATCH (p:Person {id: $applicant_id})-[r]->(e:Employer)
                RETURN type(r) as relationship, r.source as source, e.name as employer_name
                """,
                {"applicant_id": applicant_id},
            )
            
            if len(results) < 2:
                return []
            
            # Group by source
            employers_by_source = {}
            for row in results:
                source = row.get("source", "unknown")
                employers_by_source[source] = row.get("employer_name")
            
            # Compare resume vs bank_statement
            if "resume" in employers_by_source and "bank_statement" in employers_by_source:
                resume_employer = employers_by_source["resume"]
                bank_employer = employers_by_source["bank_statement"]
                
                # Check for exact match or fuzzy match (subsidiary names)
                if resume_employer.lower() not in bank_employer.lower() and bank_employer.lower() not in resume_employer.lower():
                    return [{
                        "issue_type": "employer_mismatch",
                        "severity": "NEEDS_REVIEW",
                        "field_name": "employer_name",
                        "source_doc_a": "resume",
                        "source_doc_b": "bank_statement",
                        "value_a": resume_employer,
                        "value_b": bank_employer,
                        "description": f"Employer name mismatch: Resume shows '{resume_employer}' but bank statement salary from '{bank_employer}'. Could be parent/subsidiary relationship.",
                    }]
            
            return []
            
        except Exception as e:
            logger.error(f"Employment consistency check failed: {e}")
            return []

    async def check_family_consistency(
        self,
        applicant_id: str,
        application_id: str,
    ) -> List[Dict[str, Any]]:
        """Check family size consistency across form and credit report.
        
        Args:
            applicant_id: Applicant ID
            application_id: Application ID
            
        Returns:
            List of issues found
        """
        logger.debug("[Validation] Checking family size consistency")
        
        try:
            # Get family size from application form
            app_result = await self.postgres.fetchone(
                """
                SELECT num_dependents, household_size
                FROM applications
                WHERE id = $1
                """,
                application_id,
            )
            
            if not app_result:
                return []
            
            form_dependents = app_result.get("num_dependents", 0) or 0
            
            # Get family relationships from Neo4j (from credit report or other sources)
            family_results = self.neo4j.execute_query(
                """
                MATCH (p:Person {id: $applicant_id})-[:FAMILY_OF]-(family:Person)
                RETURN count(family) as family_count
                """,
                {"applicant_id": applicant_id},
            )
            
            if not family_results:
                return []
            
            neo4j_family_count = family_results[0].get("family_count", 0)
            
            # Check for mismatch
            if form_dependents > 0 and neo4j_family_count > 0:
                if form_dependents != neo4j_family_count:
                    return [{
                        "issue_type": "family_size_mismatch",
                        "severity": "CRITICAL",
                        "field_name": "num_dependents",
                        "source_doc_a": "application_form",
                        "source_doc_b": "credit_report",
                        "value_a": str(form_dependents),
                        "value_b": str(neo4j_family_count),
                        "description": f"Family size mismatch: Form reports {form_dependents} dependents but credit report shows {neo4j_family_count} family members",
                    }]
            
            return []
            
        except Exception as e:
            logger.error(f"Family consistency check failed: {e}")
            return []

    async def perform_initial_checks(
        self,
        applicant_id: str,
        application_id: str,
    ) -> List[Dict[str, Any]]:
        """Perform initial validation checks.
        
        Args:
            applicant_id: Applicant ID
            application_id: Application ID
            
        Returns:
            List of all issues found
        """
        logger.info("[Validation] Performing initial checks")
        
        all_issues = []
        
        # Run all checks
        all_issues.extend(await self.check_address_consistency(applicant_id))
        all_issues.extend(await self.check_income_consistency(application_id))
        all_issues.extend(await self.check_employment_consistency(applicant_id))
        all_issues.extend(await self.check_family_consistency(applicant_id, application_id))
        
        logger.info(f"[Validation] Initial checks found {len(all_issues)} issues")
        
        return all_issues
    
    async def perform_partial_validation(
        self,
        extraction_results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Perform document-only validation when applicant profile unavailable.
        
        This validates data within and across documents without Neo4j queries.
        
        Args:
            extraction_results: Extracted data from all documents
            
        Returns:
            List of validation issues found
        """
        logger.info("[Validation] Performing partial validation (document-only)")
        
        issues = []
        
        # Extract data from available documents
        bank_data = extraction_results.get("bank_statement", {})
        credit_data = extraction_results.get("credit_report", {})
        resume_data = extraction_results.get("resume", {})
        
        # Check 1: Income consistency (bank vs credit)
        if bank_data and credit_data:
            bank_income = bank_data.get("total_credits", 0)
            # Credit reports sometimes have income estimates
            credit_income = credit_data.get("estimated_income", 0)
            
            if bank_income > 0 and credit_income > 0:
                diff_pct = abs(bank_income - credit_income) / bank_income
                if diff_pct > 0.25:  # 25% threshold for partial validation
                    issues.append({
                        "issue_type": "income_inconsistency_partial",
                        "severity": "NEEDS_REVIEW",
                        "field_name": "monthly_income",
                        "source_doc_a": "bank_statement",
                        "source_doc_b": "credit_report",
                        "value_a": str(bank_income),
                        "value_b": str(credit_income),
                        "description": f"Income inconsistency detected: Bank shows {bank_income} but credit report estimates {credit_income}",
                    })
        
        # Check 2: Employer name consistency (resume vs bank)
        if resume_data and bank_data:
            resume_employer = resume_data.get("current_employer", "")
            bank_employer = bank_data.get("salary_source", "")
            
            if resume_employer and bank_employer:
                if resume_employer.lower() not in bank_employer.lower() and bank_employer.lower() not in resume_employer.lower():
                    issues.append({
                        "issue_type": "employer_inconsistency_partial",
                        "severity": "NEEDS_REVIEW",
                        "field_name": "employer",
                        "source_doc_a": "resume",
                        "source_doc_b": "bank_statement",
                        "value_a": resume_employer,
                        "value_b": bank_employer,
                        "description": f"Employer name inconsistency: Resume shows '{resume_employer}' but salary from '{bank_employer}'",
                    })
        
        # Check 3: Data completeness
        required_fields = {
            "bank_statement": ["total_credits", "account_holder"],
            "credit_report": ["credit_score"],
            "resume": ["skills", "experience"],
        }
        
        for doc_type, fields in required_fields.items():
            doc_data = extraction_results.get(doc_type, {})
            if doc_data:
                for field in fields:
                    if not doc_data.get(field):
                        issues.append({
                            "issue_type": "missing_field",
                            "severity": "INFO",
                            "field_name": field,
                            "source_doc_a": doc_type,
                            "source_doc_b": None,
                            "value_a": None,
                            "value_b": None,
                            "description": f"Field '{field}' missing or empty in {doc_type}",
                        })
        
        logger.info(f"[Validation] Partial validation found {len(issues)} issues")
        return issues

    def reflect_on_findings(
        self,
        initial_issues: List[Dict[str, Any]],
        extraction_results: Dict[str, Any],
    ) -> List[str]:
        """Reflexion: Self-critique to identify missed checks.
        
        Args:
            initial_issues: Issues found in initial check
            extraction_results: All extracted data
            
        Returns:
            List of additional checks to perform
        """
        logger.info("[Reflexion] Reflecting on initial findings")
        
        # Build reflection prompt
        issues_summary = "\n".join([
            f"- {issue['issue_type']}: {issue['description']}"
            for issue in initial_issues
        ])
        
        if not issues_summary:
            issues_summary = "No issues found."
        
        available_docs = list(extraction_results.keys())
        
        reflection_prompt = f"""You are a data validation agent reviewing an applicant's documents.

Initial validation checks performed:
- Address consistency (Emirates ID vs Credit Report)
- Income consistency (Application form vs Bank Statement)
- Employment consistency (Resume vs Bank Statement)
- Family size consistency (Application form vs Credit Report)

Issues found:
{issues_summary}

Available documents:
{', '.join(available_docs)}

As an expert validator, reflect on your work:
1. Did I check all cross-document fields that could be inconsistent?
2. Are there any other validation checks I should have performed given the available documents?
3. What patterns or edge cases might I have missed?

Return a JSON list of additional checks to perform (if any):
{{
  "additional_checks": [
    "Check X between document A and B",
    "Verify Y is consistent across all sources"
  ],
  "reasoning": "Brief explanation of what you might have missed"
}}

If no additional checks are needed, return an empty list."""

        try:
            response = self.llm.generate_json(
                prompt=reflection_prompt,
                system="You are a thorough data validation expert. Identify gaps in validation coverage.",
                temperature=0.7,
            )
            
            additional_checks = response.get("additional_checks", [])
            reasoning = response.get("reasoning", "")
            
            logger.info(f"[Reflexion] LLM suggests {len(additional_checks)} additional checks")
            if reasoning:
                logger.info(f"[Reflexion] Reasoning: {reasoning}")
            
            return additional_checks
            
        except Exception as e:
            logger.error(f"Reflexion failed: {e}")
            return []

    async def store_validation_results(
        self,
        application_id: str,
        issues: List[Dict[str, Any]],
    ):
        """Store validation results in PostgreSQL.
        
        Args:
            application_id: Application ID
            issues: List of validation issues
        """
        for issue in issues:
            await self.postgres.execute(
                """
                INSERT INTO validation_results 
                (application_id, issue_type, severity, source_doc_a, source_doc_b, 
                 field_name, value_a, value_b, description)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                application_id,
                issue.get("issue_type"),
                issue.get("severity"),
                issue.get("source_doc_a"),
                issue.get("source_doc_b"),
                issue.get("field_name"),
                issue.get("value_a"),
                issue.get("value_b"),
                issue.get("description"),
            )
        
        logger.info(f"Stored {len(issues)} validation issues in PostgreSQL")


async def validate_data_agent(state: ApplicationState) -> ApplicationState:
    """Data Validation Agent node for LangGraph.
    
    Uses Reflexion pattern:
    1. Initial validation checks
    2. Self-critique reflection
    3. Revised validation (if needed)
    4. Log all issues
    
    v1: Always proceeds forward - issues flagged for human review.
    
    Args:
        state: Current application state
        
    Returns:
        Updated state with validation_issues
    """
    logger.info("=" * 80)
    logger.info("VALIDATION AGENT STARTED (Reflexion Pattern)")
    logger.info("=" * 80)
    
    state["current_agent"] = "validation"
    state["processing_status"] = "validating"
    
    # Start tracing
    trace_id = trace_agent_start(
        application_id=state["application_id"],
        agent_name="validation",
        input_state=state,
    )
    
    agent = ValidationAgent()
    
    try:
        await agent.connect_databases()
        
        applicant_id = state.get("applicant_id")
        application_id = state["application_id"]
        extraction_results = state.get("extraction_results", {})
        
        # Check which documents were successfully extracted
        available_docs = list(extraction_results.keys())
        
        if not applicant_id:
            logger.warning("No applicant_id found - attempting partial validation")
            logger.info(f"Available documents for validation: {available_docs}")
            
            # Perform document-only validation (no cross-entity checks)
            if len(available_docs) >= 2:
                logger.info("Performing partial validation on extracted documents")
                partial_issues = await agent.perform_partial_validation(extraction_results)
                
                state["validation_issues"] = partial_issues
                state["validation_passed"] = len(partial_issues) == 0
                state["validation_complete"] = True
                state["validation_notes"] = "Partial validation only (no applicant profile)"
                
                logger.info(f"Partial validation found {len(partial_issues)} issues")
                return state
            else:
                logger.warning("Insufficient documents for partial validation, skipping")
                state["validation_passed"] = True
                state["validation_complete"] = True
                state["validation_notes"] = "Skipped - insufficient data"
                return state
        
        # Initial Validation
        logger.info("[Phase 1] Initial validation checks")
        initial_issues = await agent.perform_initial_checks(applicant_id, application_id)
        
        # Reflexion
        logger.info("[Phase 2] Reflexion - self-critique")
        additional_checks = agent.reflect_on_findings(initial_issues, extraction_results)
        
        # For v1, we note the additional checks but don't implement them
        # (They would require dynamic check generation)
        if additional_checks:
            logger.info(f"[Reflexion] LLM identified {len(additional_checks)} potential additional checks:")
            for check in additional_checks:
                logger.info(f"  - {check}")
            logger.info("[v1] Additional checks noted but not implemented in v1 scope")
        
        # Store Results
        all_issues = initial_issues  # In v1, just use initial issues
        
        if all_issues:
            await agent.store_validation_results(application_id, all_issues)
            
            # Trace each validation issue
            for issue in all_issues:
                trace_validation_issue(trace_id=trace_id, issue=issue)
        
        # Update state
        state["validation_issues"] = all_issues
        state["validation_passed"] = len(all_issues) == 0
        state["validation_complete"] = True
        
        # Count by severity
        critical_count = sum(1 for i in all_issues if i.get("severity") == "CRITICAL")
        review_count = sum(1 for i in all_issues if i.get("severity") == "NEEDS_REVIEW")
        
        logger.info("=" * 80)
        logger.info(f"VALIDATION AGENT COMPLETE: {len(all_issues)} issues found")
        logger.info(f"  CRITICAL: {critical_count}")
        logger.info(f"  NEEDS_REVIEW: {review_count}")
        logger.info(f"  Status: {'PASSED' if state['validation_passed'] else 'FLAGGED FOR REVIEW'}")
        logger.info("  [v1] Proceeding to eligibility scoring (linear graph)")
        logger.info("=" * 80)
        
        # End tracing
        trace_agent_end(
            trace_id=trace_id,
            agent_name="validation",
            output_state=state,
            success=True,
        )
        
    except Exception as e:
        logger.error(f"Validation agent failed: {e}", exc_info=True)
        state["error"] = f"Validation failed: {str(e)}"
        state["processing_status"] = "error"
        # Even on error, proceed in v1
        state["validation_complete"] = True
        
        # End tracing with error
        trace_agent_end(
            trace_id=trace_id,
            agent_name="validation",
            output_state=state,
            success=False,
        )
    
    finally:
        await agent.disconnect_databases()
    
    return state


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    # Test validation agent
    from src.agents.state import create_initial_state
    
    async def test():
        state = create_initial_state("test-app-001", "APP-001")
        state["applicant_id"] = "test-applicant-uuid"
        state["extraction_results"] = {
            "emirates_id": {"status": "success"},
            "bank_statement": {"status": "success"},
            "resume": {"status": "success"},
            "credit_report": {"status": "success"},
        }
        
        result_state = await validate_data_agent(state)
        print(f"\nValidation complete: {result_state['validation_complete']}")
        print(f"Validation passed: {result_state['validation_passed']}")
        print(f"Issues found: {len(result_state.get('validation_issues', []))}")
    
    asyncio.run(test())
