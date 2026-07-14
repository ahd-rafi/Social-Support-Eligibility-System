"""Decision Recommendation Agent - ReAct pattern implementation."""

import logging
import json
from datetime import datetime
from typing import Dict, Any, List

from src.agents.state import ApplicationState
from src.agents.llm import get_llm
from src.db.postgres import PostgresClient
from src.db.mongo import MongoClient
from src.db.qdrant_client import QdrantClientWrapper
from src.ml.embeddings import get_embedding_generator
from src.observability.tracing import trace_agent_start, trace_agent_end

logger = logging.getLogger(__name__)


# LLM Prompts for Decision Reasoning

APPROVAL_PROMPT = """You are generating an approval recommendation for government social support.

Eligibility Assessment:
- Category: {category}
- Confidence: {confidence:.2f}
- Primary driving factors: {key_factors}

Applicant Profile Summary:
{profile_summary}

Validation Issues:
{validation_summary}

Generate a clear, empathetic approval recommendation.

Return JSON:
{{
  "decision": "approved",
  "reasoning": "3-4 sentences explaining why this applicant qualifies, citing the 2-3 most important factors from the assessment",
  "key_factors": ["Top 3 factors that support this decision"],
  "confidence": {confidence},
  "review_notes": "Any caveats or items for case worker review (or empty string if none)"
}}

Be specific about financial circumstances. Acknowledge any validation issues flagged."""

DECLINE_PROMPT = """You are generating a soft decline recommendation for government social support.

Eligibility Assessment:
- Category: {category}
- Confidence: {confidence:.2f}
- Primary factors: {key_factors}

Applicant Profile Summary:
{profile_summary}

Validation Issues:
{validation_summary}

Generate a respectful soft decline recommendation with clear reasoning.

Return JSON:
{{
  "decision": "soft_decline",
  "reasoning": "3-4 sentences explaining why the applicant does not currently qualify, citing specific factors",
  "key_factors": ["Top 3 factors that led to this decision"],
  "confidence": {confidence},
  "review_notes": "What would need to change for the applicant to qualify in the future"
}}

Be empathetic but clear. Provide actionable guidance."""


class RecommendationAgent:
    """Decision Recommendation Agent using ReAct pattern.
    
    Workflow:
    1. Review eligibility score, feature importances, validation issues
    2. Generate approval/decline reasoning using LLM
    3. Query Qdrant for semantically similar enablement programs
    4. Compile final recommendation with key factors and review notes
    5. Store decision in PostgreSQL audit trail
    """

    def __init__(self):
        """Initialize recommendation agent."""
        self.llm = get_llm()
        self.postgres = PostgresClient()
        self.mongo = MongoClient()
        self.qdrant = QdrantClientWrapper()
        self.embedder = get_embedding_generator()
        
        logger.info("Recommendation agent initialized")

    async def connect_databases(self):
        """Connect to databases."""
        await self.postgres.connect()
        self.mongo.connect()
        self.qdrant.connect()

    async def disconnect_databases(self):
        """Disconnect from databases."""
        await self.postgres.disconnect()
        self.mongo.disconnect()
        self.qdrant.disconnect()

    def generate_profile_summary(
        self,
        features: Dict[str, Any],
    ) -> str:
        """Generate human-readable profile summary.
        
        Args:
            features: Feature vector
            
        Returns:
            Profile summary string
        """
        monthly_income = features.get("monthly_income", 0)
        household_size = features.get("household_size", 1)
        income_per_family = features.get("income_per_family", 0)
        net_worth = features.get("net_worth", 0)
        credit_score = features.get("credit_score", 0)
        employment_months = features.get("employment_months", 0)
        
        summary = f"""
- Monthly income: {monthly_income:.2f} AED
- Household size: {household_size}
- Income per family member: {income_per_family:.2f} AED
- Net worth: {net_worth:.2f} AED
- Credit score: {credit_score}
- Employment duration: {employment_months} months
"""
        return summary.strip()

    def generate_decision_reasoning(
        self,
        category: str,
        confidence: float,
        features: Dict[str, Any],
        importances: Dict[str, float],
        validation_issues: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate decision reasoning using LLM.
        
        Args:
            category: Eligibility category
            confidence: Confidence score
            features: Feature vector
            importances: Feature importances
            validation_issues: Validation issues
            
        Returns:
            Decision reasoning dict
        """
        logger.debug("[ReAct] Action: Generating decision reasoning with LLM")
        
        # Format key factors (top 3 importances)
        top_factors = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]
        key_factors_str = ", ".join([
            f"{factor} ({importance:.2f})"
            for factor, importance in top_factors
        ])
        
        # Generate profile summary
        profile_summary = self.generate_profile_summary(features)
        
        # Format validation issues
        if validation_issues:
            validation_summary = "\n".join([
                f"  [{issue.get('severity')}] {issue.get('description')}"
                for issue in validation_issues
            ])
        else:
            validation_summary = "  No issues detected"
        
        # Select prompt based on category
        if category == "soft_decline":
            prompt_template = DECLINE_PROMPT
        else:
            prompt_template = APPROVAL_PROMPT
        
        prompt = prompt_template.format(
            category=category,
            confidence=confidence,
            key_factors=key_factors_str,
            profile_summary=profile_summary,
            validation_summary=validation_summary,
        )
        
        try:
            reasoning = self.llm.generate_json(
                prompt=prompt,
                system="You are an expert social support case worker. Generate clear, empathetic recommendations.",
                temperature=0.7,
            )
            
            logger.debug("[ReAct] Observation: Decision reasoning generated")
            return reasoning
            
        except Exception as e:
            logger.error(f"Failed to generate reasoning: {e}")
            return {
                "decision": "approved" if category != "soft_decline" else "soft_decline",
                "reasoning": f"Eligibility category: {category} with {confidence:.2f} confidence",
                "key_factors": [f"{k}" for k, v in list(importances.items())[:3]],
                "confidence": confidence,
                "review_notes": "LLM reasoning unavailable - manual review recommended",
            }

    async def match_enablement_programs(
        self,
        application_id: str,
        min_score: float = 0.70,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Match enablement programs using Qdrant semantic search with keyword fallback.
        
        Args:
            application_id: Application ID
            min_score: Minimum similarity score threshold
            limit: Maximum number of programs to return
            
        Returns:
            List of matched programs
        """
        logger.debug("[ReAct] Action: Matching enablement programs")
        
        try:
            # Get applicant profile from extractions
            extraction_result = await self.postgres.fetchone(
                """
                SELECT extraction_data
                FROM extraction_results
                WHERE application_id = $1 AND document_type = 'resume'
                """,
                application_id,
            )
            
            if not extraction_result:
                logger.warning("No resume found in PostgreSQL - trying MongoDB fallback")
                # Try MongoDB as fallback
                programs = await self.match_programs_keyword_fallback(application_id, limit)
                return programs
            
            extraction_data = extraction_result.get("extraction_data", {})
            
            # Build text for embedding
            skills = extraction_data.get("skills", [])
            experience = extraction_data.get("experience", [])
            
            # Create text representation
            skills_text = " ".join(skills) if skills else ""
            exp_text = " ".join([
                f"{exp.get('title', '')} at {exp.get('employer', '')}"
                for exp in experience
            ]) if experience else ""
            
            query_text = f"{skills_text} {exp_text}"
            
            if not query_text.strip():
                logger.warning("No skills or experience data - trying keyword fallback")
                return await self.match_programs_keyword_fallback(application_id, limit)
            
            # Generate embedding
            query_embedding = self.embedder.embed(query_text)
            
            # Search Qdrant
            results = self.qdrant.search(
                collection_name="enablement_programs",
                query_vector=query_embedding,
                limit=limit,
                score_threshold=min_score,
            )
            
            # Format results
            programs = []
            for result in results:
                payload = result["payload"]
                programs.append({
                    "program_name": payload.get("name", "Unknown Program"),
                    "match_score": result["score"],
                    "description": payload.get("description", ""),
                    "provider": payload.get("provider", ""),
                    "duration": payload.get("duration", ""),
                    "category": payload.get("category", ""),
                    "match_method": "semantic_embedding",
                })
            
            if not programs:
                logger.warning("No programs matched with embeddings - trying keyword fallback")
                return await self.match_programs_keyword_fallback(application_id, limit)
            
            logger.info(f"[ReAct] Observation: Matched {len(programs)} programs (semantic, score >= {min_score})")
            return programs
            
        except Exception as e:
            logger.error(f"Program matching failed: {e}, trying keyword fallback", exc_info=True)
            return await self.match_programs_keyword_fallback(application_id, limit)
    
    async def match_programs_keyword_fallback(
        self,
        application_id: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Fallback keyword-based program matching when embeddings unavailable.
        
        Args:
            application_id: Application ID
            limit: Maximum programs to return
            
        Returns:
            List of matched programs with lower confidence scores
        """
        logger.info("[Fallback] Using keyword-based program matching")
        
        try:
            # Try to get skills from MongoDB extractions
            extraction_doc = self.mongo.find_one(
                "extractions",
                {"application_id": application_id, "document_type": "resume"}
            )
            
            if not extraction_doc:
                logger.warning("No resume data in MongoDB either")
                return []
            
            structured_data = extraction_doc.get("structured_data", {})
            skills = structured_data.get("skills", [])
            sector = structured_data.get("sector", "")
            
            if not skills and not sector:
                logger.warning("No skills or sector data for keyword matching")
                return []
            
            # Simple keyword matching against program database
            # In production, this would query a programs table
            # For now, return generic recommendations based on sector
            
            sector_programs = {
                "retail": [
                    {"name": "Customer Service Excellence", "category": "upskilling", "match": 0.65},
                    {"name": "Retail Management Training", "category": "upskilling", "match": 0.60},
                    {"name": "Sales Techniques Workshop", "category": "training", "match": 0.55},
                ],
                "hospitality": [
                    {"name": "Hospitality Management Program", "category": "upskilling", "match": 0.65},
                    {"name": "Food Service Training", "category": "training", "match": 0.60},
                ],
                "construction": [
                    {"name": "Construction Safety Certification", "category": "certification", "match": 0.65},
                    {"name": "Project Management Basics", "category": "upskilling", "match": 0.60},
                ],
                "healthcare": [
                    {"name": "Healthcare Assistant Training", "category": "certification", "match": 0.65},
                    {"name": "Medical Terminology Course", "category": "training", "match": 0.60},
                ],
            }
            
            # Generic programs for all sectors
            generic_programs = [
                {"name": "Financial Literacy Workshop", "category": "life_skills", "match": 0.50},
                {"name": "Job Search Skills", "category": "employment_support", "match": 0.50},
                {"name": "Resume Writing Workshop", "category": "employment_support", "match": 0.45},
            ]
            
            matched_programs = []
            
            # Add sector-specific programs
            sector_lower = sector.lower() if sector else ""
            for key, programs in sector_programs.items():
                if key in sector_lower or key in " ".join(skills).lower():
                    matched_programs.extend(programs)
                    break
            
            # Add generic programs
            matched_programs.extend(generic_programs)
            
            # Format and limit
            result = []
            for prog in matched_programs[:limit]:
                result.append({
                    "program_name": prog["name"],
                    "match_score": prog["match"],
                    "description": f"Recommended based on sector/skills match",
                    "provider": "Government Programs",
                    "duration": "Varies",
                    "category": prog["category"],
                    "match_method": "keyword_fallback",
                })
            
            logger.info(f"[Fallback] Matched {len(result)} programs using keywords")
            return result
            
        except Exception as e:
            logger.error(f"Keyword fallback matching failed: {e}")
            return []

    async def store_decision(
        self,
        application_id: str,
        recommendation: Dict[str, Any],
        enablement_programs: List[Dict[str, Any]],
    ):
        """Store final decision in PostgreSQL audit trail.
        
        Args:
            application_id: Application ID
            recommendation: Decision recommendation
            enablement_programs: Matched programs
        """
        await self.postgres.execute(
            """
            INSERT INTO decisions 
            (application_id, decision, confidence, reasoning, key_factors, 
             enablement_programs, review_notes)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (application_id) DO UPDATE SET
                decision = EXCLUDED.decision,
                confidence = EXCLUDED.confidence,
                reasoning = EXCLUDED.reasoning,
                key_factors = EXCLUDED.key_factors,
                enablement_programs = EXCLUDED.enablement_programs,
                review_notes = EXCLUDED.review_notes,
                decided_at = CURRENT_TIMESTAMP
            """,
            application_id,
            recommendation.get("decision"),
            recommendation.get("confidence"),
            recommendation.get("reasoning"),
            json.dumps(recommendation.get("key_factors", [])),
            json.dumps(enablement_programs),
            recommendation.get("review_notes", ""),
        )
        
        # Update application status
        await self.postgres.execute(
            """
            UPDATE applications
            SET status = 'completed', completed_at = CURRENT_TIMESTAMP
            WHERE id = $1
            """,
            application_id,
        )
        
        logger.info(f"Stored decision in PostgreSQL audit trail")


async def decision_recommendation_agent(state: ApplicationState) -> ApplicationState:
    """Decision Recommendation Agent node for LangGraph.
    
    ReAct workflow:
    1. Review eligibility score and validation issues
    2. Generate approval/decline reasoning with LLM
    3. Match enablement programs via Qdrant semantic search
    4. Compile final recommendation
    5. Store in PostgreSQL audit trail
    
    Args:
        state: Current application state
        
    Returns:
        Updated state with recommendation and enablement_programs
    """
    logger.info("=" * 80)
    logger.info("DECISION RECOMMENDATION AGENT STARTED (ReAct Pattern)")
    logger.info("=" * 80)
    
    state["current_agent"] = "recommendation"
    state["processing_status"] = "recommending"
    
    # Start tracing
    trace_id = trace_agent_start(
        application_id=state["application_id"],
        agent_name="recommendation",
        input_state=state,
    )
    
    agent = RecommendationAgent()
    
    try:
        await agent.connect_databases()
        
        application_id = state["application_id"]
        category = state.get("eligibility_category", "unknown")
        confidence = state.get("eligibility_score", 0.0)
        features = state.get("feature_vector", {})
        importances = state.get("feature_importances", {})
        validation_issues = state.get("validation_issues", [])
        
        # Step 1: Generate decision reasoning
        logger.info(f"[ReAct] Thought: Eligibility category is '{category}' with {confidence:.2f} confidence")
        logger.info("[ReAct] Action: Generating decision reasoning")
        
        recommendation = agent.generate_decision_reasoning(
            category, confidence, features, importances, validation_issues
        )
        
        logger.info(f"[ReAct] Observation: Decision reasoning complete")
        logger.info(f"[ReAct]   Decision: {recommendation.get('decision')}")
        logger.info(f"[ReAct]   Reasoning: {recommendation.get('reasoning')[:100]}...")
        
        # Step 2: Match enablement programs
        logger.info("[ReAct] Thought: I need to match enablement programs for this applicant")
        logger.info("[ReAct] Action: Querying Qdrant for semantic program matches")
        
        enablement_programs = await agent.match_enablement_programs(application_id)
        
        if enablement_programs:
            logger.info(f"[ReAct] Observation: Matched {len(enablement_programs)} programs:")
            for prog in enablement_programs[:3]:  # Log top 3
                logger.info(f"[ReAct]   - {prog['program_name']} (score: {prog['match_score']:.3f})")
        else:
            logger.warning("[ReAct] Observation: No enablement programs matched")
        
        # Step 3: Store decision
        logger.info("[ReAct] Thought: I have everything. I'll store the final decision.")
        await agent.store_decision(application_id, recommendation, enablement_programs)
        logger.info("[ReAct] Observation: Decision stored in audit trail")
        
        # Update state
        state["recommendation"] = recommendation
        state["enablement_programs"] = enablement_programs
        state["recommendation_complete"] = True
        state["processing_status"] = "complete"
        state["completed_at"] = datetime.utcnow().isoformat()
        
        logger.info("=" * 80)
        logger.info(f"DECISION RECOMMENDATION COMPLETE")
        logger.info(f"  Final Decision: {recommendation.get('decision')}")
        logger.info(f"  Confidence: {recommendation.get('confidence'):.3f}")
        logger.info(f"  Enablement Programs: {len(enablement_programs)}")
        logger.info(f"  Review Notes: {recommendation.get('review_notes') or 'None'}")
        logger.info("=" * 80)
        logger.info("")
        logger.info("🎉 APPLICATION PROCESSING COMPLETE 🎉")
        logger.info("")
        
        # End tracing
        trace_agent_end(
            trace_id=trace_id,
            agent_name="recommendation",
            output_state=state,
            success=True,
        )
        
    except Exception as e:
        logger.error(f"Recommendation agent failed: {e}", exc_info=True)
        state["error"] = f"Recommendation failed: {str(e)}"
        state["processing_status"] = "error"
        
        # End tracing with error
        trace_agent_end(
            trace_id=trace_id,
            agent_name="recommendation",
            output_state=state,
            success=False,
        )
    
    finally:
        await agent.disconnect_databases()
    
    return state


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    # Test recommendation agent
    from src.agents.state import create_initial_state
    
    async def test():
        state = create_initial_state("test-app-001", "APP-001")
        state["eligibility_category"] = "approved_financial"
        state["eligibility_score"] = 0.84
        state["feature_vector"] = {
            "monthly_income": 8000,
            "household_size": 5,
            "income_per_family": 1600,
            "net_worth": 5000,
            "credit_score": 620,
            "employment_months": 36,
        }
        state["feature_importances"] = {
            "income_per_family": 0.34,
            "net_worth": 0.22,
            "credit_score": 0.18,
        }
        state["validation_issues"] = []
        
        result_state = await decision_recommendation_agent(state)
        print(f"\nRecommendation complete: {result_state.get('recommendation_complete')}")
        print(f"Decision: {result_state.get('recommendation', {}).get('decision')}")
        print(f"Programs matched: {len(result_state.get('enablement_programs', []))}")
    
    asyncio.run(test())
