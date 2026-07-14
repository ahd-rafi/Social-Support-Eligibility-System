"""Eligibility Check Agent - ReAct pattern implementation."""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from src.agents.state import ApplicationState
from src.agents.llm import get_llm
from src.db.postgres import PostgresClient
from src.ml.classifier import get_classifier
from src.observability.tracing import trace_agent_start, trace_agent_end, trace_classifier_prediction

logger = logging.getLogger(__name__)


# LLM Prompt for Score Interpretation

INTERPRETATION_PROMPT = """You are an expert social support case worker for Abu Dhabi government services.

The eligibility classifier has predicted:
Category: {category}
Confidence: {confidence:.2f}

Feature values for this applicant:
{features_formatted}

Feature importances (what drove this decision):
{importances_formatted}

Validation issues detected during document review:
{validation_issues_summary}

Your task:
1. Interpret this eligibility score in the context of the applicant's full situation
2. Identify any edge cases or red flags (e.g., very recent employment, major inconsistencies, unusual financial patterns)
3. Determine if a human reviewer should pay special attention to anything

Return JSON:
{{
  "interpretation": "1-2 sentence summary explaining the decision in plain language",
  "edge_cases": [
    "List any notable points that might affect the decision",
    "E.g., 'Recently started employment (only 4 months)'",
    "E.g., 'Income reported on form significantly differs from bank statement'"
  ],
  "review_recommended": true or false,
  "review_reason": "Why human review is recommended (if applicable)",
  "reasoning": "Detailed explanation connecting the features, importances, and validation issues to the final category"
}}

Focus on practical case management. Be specific about what factors matter."""


class EligibilityAgent:
    """Eligibility Check Agent using ReAct pattern.
    
    Workflow:
    1. Assemble 12-feature vector from PostgreSQL
    2. Run HistGradientBoostingClassifier
    3. Get feature importances (for explainability)
    4. LLM interprets the score in context (validation issues, edge cases)
    """

    def __init__(self):
        """Initialize eligibility agent."""
        self.llm = get_llm()
        self.classifier = get_classifier()
        self.postgres = PostgresClient()
        
        logger.info("Eligibility agent initialized")

    async def connect_database(self):
        """Connect to PostgreSQL."""
        await self.postgres.connect()

    async def disconnect_database(self):
        """Disconnect from PostgreSQL."""
        await self.postgres.disconnect()

    async def assemble_feature_vector(
        self,
        application_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Assemble 12-feature vector from PostgreSQL.
        
        Features:
        - monthly_income
        - employment_months
        - household_size
        - num_dependents
        - income_per_family
        - total_assets
        - total_liabilities
        - net_worth
        - debt_to_income
        - credit_score
        - missed_payments
        - age
        
        Args:
            application_id: Application ID
            
        Returns:
            Feature vector as dict, or None if incomplete
        """
        logger.debug("[ReAct] Action: Assembling feature vector from PostgreSQL")
        
        try:
            # Get all relevant data in one query
            query = """
            SELECT 
                a.age,
                app.monthly_income,
                app.household_size,
                app.num_dependents,
                f.employment_months,
                f.employment_status,
                f.total_assets,
                f.total_liabilities,
                f.net_worth,
                f.debt_to_income,
                f.credit_score,
                f.missed_payments
            FROM applications app
            LEFT JOIN applicants a ON app.applicant_id = a.id
            LEFT JOIN applicant_features f ON app.id = f.application_id
            WHERE app.id = $1
            """
            
            result = await self.postgres.fetchone(query, application_id)
            
            if not result:
                logger.error(f"No data found for application {application_id}")
                return None
            
            # Extract values
            monthly_income = float(result.get("monthly_income") or 0)
            household_size = int(result.get("household_size") or 1)
            num_dependents = int(result.get("num_dependents") or 0)
            
            # Calculate derived features
            income_per_family = monthly_income / household_size if household_size > 0 else monthly_income
            
            debt_to_income = result.get("debt_to_income")
            if debt_to_income is None:
                # Calculate if not already computed
                total_liabilities = float(result.get("total_liabilities") or 0)
                annual_income = monthly_income * 12
                debt_to_income = total_liabilities / annual_income if annual_income > 0 else 0.0
            else:
                debt_to_income = float(debt_to_income)
            
            features = {
                "monthly_income": monthly_income,
                "employment_months": int(result.get("employment_months") or 0),
                "household_size": household_size,
                "num_dependents": num_dependents,
                "income_per_family": income_per_family,
                "total_assets": float(result.get("total_assets") or 0),
                "total_liabilities": float(result.get("total_liabilities") or 0),
                "net_worth": float(result.get("net_worth") or 0),
                "debt_to_income": debt_to_income,
                "credit_score": int(result.get("credit_score") or 0),
                "missed_payments": int(result.get("missed_payments") or 0),
                "age": int(result.get("age") or 0),
            }
            
            logger.debug(f"[ReAct] Observation: Assembled {len(features)} features")
            return features
            
        except Exception as e:
            logger.error(f"Failed to assemble feature vector: {e}", exc_info=True)
            return None

    def interpret_with_llm(
        self,
        category: str,
        confidence: float,
        features: Dict[str, Any],
        importances: Dict[str, float],
        validation_issues: list,
    ) -> Dict[str, Any]:
        """Interpret eligibility score using LLM.
        
        Args:
            category: Predicted category
            confidence: Confidence score
            features: Feature vector
            importances: Feature importances
            validation_issues: List of validation issues
            
        Returns:
            Interpretation dict
        """
        logger.debug("[ReAct] Action: Interpreting score with LLM")
        
        # Format features for prompt
        features_formatted = "\n".join([
            f"  {key}: {value}"
            for key, value in features.items()
        ])
        
        # Format importances (top 5)
        top_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
        importances_formatted = "\n".join([
            f"  {key}: {value:.3f} (importance)"
            for key, value in top_importances
        ])
        
        # Format validation issues
        if validation_issues:
            validation_issues_summary = "\n".join([
                f"  [{issue.get('severity')}] {issue.get('description')}"
                for issue in validation_issues
            ])
        else:
            validation_issues_summary = "  No validation issues detected"
        
        # Build prompt
        prompt = INTERPRETATION_PROMPT.format(
            category=category,
            confidence=confidence,
            features_formatted=features_formatted,
            importances_formatted=importances_formatted,
            validation_issues_summary=validation_issues_summary,
        )
        
        try:
            interpretation = self.llm.generate_json(
                prompt=prompt,
                system="You are an expert social support case worker. Provide clear, practical interpretation of eligibility decisions.",
                temperature=0.7,
            )
            
            logger.debug("[ReAct] Observation: LLM interpretation complete")
            return interpretation
            
        except Exception as e:
            logger.error(f"LLM interpretation failed: {e}")
            return {
                "interpretation": f"Predicted {category} with {confidence:.2f} confidence",
                "edge_cases": [],
                "review_recommended": False,
                "reasoning": "LLM interpretation unavailable",
            }

    async def store_eligibility_score(
        self,
        application_id: str,
        score: float,
        category: str,
        importances: Dict[str, float],
        interpretation: Dict[str, Any],
    ):
        """Store eligibility score in PostgreSQL.
        
        Args:
            application_id: Application ID
            score: Confidence score (0-1)
            category: Predicted category
            importances: Feature importances
            interpretation: LLM interpretation
        """
        import json
        
        notes = f"{interpretation.get('interpretation', '')}. {interpretation.get('reasoning', '')}"
        
        await self.postgres.execute(
            """
            INSERT INTO eligibility_scores 
            (application_id, score, category, feature_importances, notes)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (application_id) DO UPDATE SET
                score = EXCLUDED.score,
                category = EXCLUDED.category,
                feature_importances = EXCLUDED.feature_importances,
                notes = EXCLUDED.notes,
                scored_at = CURRENT_TIMESTAMP
            """,
            application_id,
            score,
            category,
            json.dumps(importances),
            notes,
        )
        
        logger.info(f"Stored eligibility score in PostgreSQL")


async def eligibility_check_agent(state: ApplicationState) -> ApplicationState:
    """Eligibility Check Agent node for LangGraph.
    
    ReAct workflow:
    1. Assemble feature vector from PostgreSQL
    2. Run HistGradientBoostingClassifier
    3. Get feature importances
    4. LLM interprets score in context
    5. Store results
    
    Args:
        state: Current application state
        
    Returns:
        Updated state with eligibility_score, category, importances
    """
    logger.info("=" * 80)
    logger.info("ELIGIBILITY CHECK AGENT STARTED (ReAct Pattern)")
    logger.info("=" * 80)
    
    state["current_agent"] = "eligibility"
    state["processing_status"] = "scoring"
    
    # Start tracing
    trace_id = trace_agent_start(
        application_id=state["application_id"],
        agent_name="eligibility",
        input_state=state,
    )
    
    agent = EligibilityAgent()
    
    try:
        await agent.connect_database()
        
        application_id = state["application_id"]
        validation_issues = state.get("validation_issues", [])
        
        # Step 1: Assemble features
        logger.info("[ReAct] Thought: I need to assemble the feature vector")
        features = await agent.assemble_feature_vector(application_id)
        
        if not features:
            logger.error("Failed to assemble feature vector - cannot score eligibility")
            state["error"] = "Incomplete feature vector"
            state["processing_status"] = "error"
            return state
        
        logger.info(f"[ReAct] Observation: Feature vector complete with {len(features)} features")
        
        # Step 2: Run classifier
        logger.info("[ReAct] Thought: I have features. I'll run the classifier.")
        
        if not agent.classifier.is_trained:
            logger.warning("Classifier not trained - loading model")
            try:
                agent.classifier.load()
            except Exception as e:
                logger.error(f"Failed to load classifier: {e}")
                state["error"] = "Classifier model not available"
                state["processing_status"] = "error"
                return state
        
        prediction = agent.classifier.predict(features)
        
        category = prediction["category"]
        confidence = prediction["confidence"]
        importances = prediction["feature_importances"]
        
        logger.info(f"[ReAct] Observation: Classifier prediction: {category} ({confidence:.3f})")
        
        # Trace classifier prediction
        trace_classifier_prediction(
            trace_id=trace_id,
            features=features,
            prediction={
                "category": category,
                "confidence": confidence,
                "importances": importances,
            },
        )
        
        # Step 3: LLM Interpretation
        logger.info("[ReAct] Thought: I should interpret this score in context")
        interpretation = agent.interpret_with_llm(
            category, confidence, features, importances, validation_issues
        )
        
        logger.info(f"[ReAct] Observation: {interpretation.get('interpretation')}")
        if interpretation.get("edge_cases"):
            logger.info(f"[ReAct] Edge cases identified: {interpretation['edge_cases']}")
        
        # Step 4: Store results
        await agent.store_eligibility_score(
            application_id, confidence, category, importances, interpretation
        )
        
        # Update state
        state["feature_vector"] = features
        state["eligibility_score"] = confidence
        state["eligibility_category"] = category
        state["feature_importances"] = importances
        state["eligibility_complete"] = True
        
        # Store interpretation notes in state for recommendation agent
        state["eligibility_interpretation"] = interpretation
        
        # Determine review flag
        review_recommended = interpretation.get("review_recommended", False)
        
        logger.info("=" * 80)
        logger.info(f"ELIGIBILITY CHECK COMPLETE")
        logger.info(f"  Category: {category}")
        logger.info(f"  Confidence: {confidence:.3f}")
        logger.info(f"  Top driving factor: {list(importances.keys())[0]} ({list(importances.values())[0]:.3f})")
        logger.info(f"  Review recommended: {review_recommended}")
        logger.info("=" * 80)
        
        # End tracing
        trace_agent_end(
            trace_id=trace_id,
            agent_name="eligibility",
            output_state=state,
            success=True,
        )
        
    except Exception as e:
        logger.error(f"Eligibility check agent failed: {e}", exc_info=True)
        state["error"] = f"Eligibility check failed: {str(e)}"
        state["processing_status"] = "error"
        
        # End tracing with error
        trace_agent_end(
            trace_id=trace_id,
            agent_name="eligibility",
            output_state=state,
            success=False,
        )
    
    finally:
        await agent.disconnect_database()
    
    return state


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    # Test eligibility agent
    from src.agents.state import create_initial_state
    
    async def test():
        state = create_initial_state("test-app-001", "APP-001")
        state["validation_issues"] = []
        
        # Note: This test requires actual data in PostgreSQL
        result_state = await eligibility_check_agent(state)
        print(f"\nEligibility check complete: {result_state.get('eligibility_complete')}")
        print(f"Category: {result_state.get('eligibility_category')}")
        print(f"Score: {result_state.get('eligibility_score')}")
    
    asyncio.run(test())
