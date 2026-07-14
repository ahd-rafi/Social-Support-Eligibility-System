"""Data Extraction Agent - ReAct pattern implementation."""

import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.agents.state import ApplicationState
from src.agents.llm import get_llm
from src.parsers.router import get_parser_router, DocumentType
from src.db.postgres import PostgresClient
from src.db.mongo import MongoClient
from src.db.neo4j_client import Neo4jClient
from src.db.qdrant_client import QdrantClientWrapper
from src.ml.embeddings import get_embedding_generator
from src.observability.tracing import trace_agent_start, trace_agent_end, trace_llm_call

logger = logging.getLogger(__name__)


# LLM Prompts for Structuring Extracted Data

EMIRATES_ID_PROMPT = """Extract structured data from this Emirates ID OCR text.

CRITICAL: You MUST extract the name field. Look carefully for "Name" label and the value after it.

OCR Text:
{raw_text}

Return valid JSON with exactly these fields:
{{
  "name": "Full name in English (REQUIRED - extract from 'Name' or 'Name (English)' field)",
  "id_number": "Emirates ID number (784-YYYY-NNNNNNN-C format)",
  "dob": "Date of birth (YYYY-MM-DD format)",
  "nationality": "Nationality (UAE/Other)",
  "gender": "M or F",
  "expiry_date": "ID expiry date (YYYY-MM-DD format)"
}}

IMPORTANT:
- name field is REQUIRED - look for it carefully in the text
- If you see "Name (English):" look at the next line for the actual name
- If field cannot be found, use null
- Return ONLY valid JSON, no markdown, no explanations"""

BANK_STATEMENT_PROMPT = """Extract structured data from this bank statement text.

Statement Text:
{raw_text}

Return valid JSON with these fields:
{{
  "account_holder": "Account holder name",
  "bank_name": "Bank name",
  "period_start": "Statement period start (YYYY-MM-DD)",
  "period_end": "Statement period end (YYYY-MM-DD)",
  "opening_balance": 0.00,
  "closing_balance": 0.00,
  "total_credits": 0.00,
  "total_debits": 0.00,
  "average_balance": 0.00,
  "salary_deposits": [
    {{"date": "YYYY-MM-DD", "amount": 0.00, "source": "Employer name"}}
  ],
  "transactions": [
    {{"date": "YYYY-MM-DD", "description": "Transaction description", "amount": 0.00, "balance": 0.00}}
  ]
}}

Extract key insights: total monthly income from salary deposits, employer name from salary source.
Return only JSON, no additional text."""

RESUME_PROMPT = """Extract structured data from this resume text.

Resume Text:
{raw_text}

Return valid JSON with these fields:
{{
  "name": "Applicant name",
  "contact": {{
    "email": "email@example.com",
    "phone": "+971...",
    "emirate": "Emirate"
  }},
  "professional_summary": "Brief summary",
  "experience": [
    {{
      "employer": "Company name",
      "title": "Job title",
      "start_date": "YYYY-MM",
      "end_date": "YYYY-MM or Present",
      "duration_months": 0,
      "description": "Role description"
    }}
  ],
  "education": [
    {{
      "institution": "University/School name",
      "degree": "Degree/Certification",
      "year": "YYYY"
    }}
  ],
  "skills": ["skill1", "skill2", "skill3"]
}}

Calculate duration_months for each job. Extract all skills mentioned.
Return only JSON, no additional text."""

CREDIT_REPORT_PROMPT = """Extract structured data from this credit report text.

Credit Report Text:
{raw_text}

Return valid JSON with these fields:
{{
  "credit_score": 650,
  "score_range": "300-900",
  "report_date": "YYYY-MM-DD",
  "applicant_name": "Name",
  "address": {{
    "street": "Street",
    "emirate": "Emirate",
    "country": "UAE"
  }},
  "employment": {{
    "employer": "Company name",
    "position": "Job title",
    "monthly_income": 0.00
  }},
  "accounts": [
    {{
      "creditor": "Bank/Lender name",
      "account_type": "Credit card/Loan/Mortgage",
      "credit_limit": 0.00,
      "balance": 0.00,
      "monthly_payment": 0.00,
      "status": "Current/Delinquent"
    }}
  ],
  "total_outstanding_debt": 0.00,
  "payment_history": {{
    "on_time_payments_pct": 0.0,
    "missed_payments_count": 0,
    "recent_missed_payments": 0
  }}
}}

Extract all debt accounts and payment history details.
Return only JSON, no additional text."""


class ExtractionAgent:
    """Data Extraction Agent using ReAct pattern.
    
    Processes uploaded documents:
    1. Parse each document using appropriate parser (OCR, PDF, Excel)
    2. Structure raw data using LLM with document-specific prompts
    3. Store across databases:
       - MongoDB: Raw extraction data (full JSON)
       - PostgreSQL: Structured applicant profile and features
       - Neo4j: Entity graph (Person, Employer, Address nodes)
       - Qdrant: Resume embeddings for program matching
    """

    def __init__(self):
        """Initialize extraction agent."""
        self.llm = get_llm()
        self.parser = get_parser_router()
        self.embedder = get_embedding_generator()
        
        # Database clients
        self.postgres = PostgresClient()
        self.mongo = MongoClient()
        self.neo4j = Neo4jClient()
        self.qdrant = QdrantClientWrapper()
        
        logger.info("Extraction agent initialized")

    async def connect_databases(self):
        """Connect to all databases."""
        await self.postgres.connect()
        self.mongo.connect()
        self.neo4j.connect()
        self.qdrant.connect()

    async def disconnect_databases(self):
        """Disconnect from all databases."""
        await self.postgres.disconnect()
        self.mongo.disconnect()
        self.neo4j.disconnect()
        self.qdrant.disconnect()

    def structure_with_llm(
        self,
        raw_text: str,
        document_type: DocumentType,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Structure raw text using LLM with document-specific prompt.
        
        Args:
            raw_text: Raw extracted text
            document_type: Type of document
            
        Returns:
            Structured data as dict
        """
        # Select prompt based on document type
        prompts = {
            DocumentType.EMIRATES_ID: EMIRATES_ID_PROMPT,
            DocumentType.BANK_STATEMENT: BANK_STATEMENT_PROMPT,
            DocumentType.RESUME: RESUME_PROMPT,
            DocumentType.CREDIT_REPORT: CREDIT_REPORT_PROMPT,
        }
        
        prompt_template = prompts.get(document_type)
        if not prompt_template:
            logger.warning(f"No LLM prompt for document type: {document_type}")
            return {"raw_text": raw_text}
        
        prompt = prompt_template.format(raw_text=raw_text)
        
        try:
            # Call LLM with JSON mode
            structured_data = self.llm.generate_json(
                prompt=prompt,
                system="You are a data extraction assistant. Extract structured data from documents and return valid JSON only.",
                temperature=0.3,  # Lower temperature for more consistent extraction
            )
            
            # Trace LLM call
            if trace_id:
                trace_llm_call(
                    trace_id=trace_id,
                    model=self.llm.model,  # Fixed: use .model not .model_name
                    prompt=prompt[:500],  # Truncate for readability
                    completion=json.dumps(structured_data)[:500],
                    metadata={"document_type": document_type.value, "temperature": 0.3},
                )
            
            logger.info(f"✓ Structured {document_type.value} with LLM")
            return structured_data
            
        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON for {document_type.value}: {e}")
            # Retry with more explicit prompt
            retry_prompt = prompt + "\n\nIMPORTANT: Return ONLY valid JSON. No markdown, no explanation."
            try:
                structured_data = self.llm.generate_json(
                    prompt=retry_prompt,
                    temperature=0.1,
                )
                logger.info(f"✓ Structured {document_type.value} on retry")
                return structured_data
            except:
                logger.error(f"Retry failed for {document_type.value}, storing raw text")
                return {"raw_text": raw_text, "error": "Failed to structure"}

    async def store_emirates_id(
        self,
        application_id: str,
        data: Dict[str, Any],
        raw_text: str,
    ) -> str:
        """Store Emirates ID data across databases.
        
        Returns:
            Applicant ID (UUID)
        """
        # Store in MongoDB (raw)
        mongo_doc = {
            "application_id": application_id,
            "document_type": "emirates_id",
            "raw_text": raw_text,
            "structured_data": data,
            "extracted_at": datetime.utcnow().isoformat(),
        }
        self.mongo.insert_one("extractions", mongo_doc)
        
        # Calculate age from DOB
        age = None
        dob_obj = None
        if data.get("dob"):
            try:
                from datetime import datetime as dt
                dob_str = data["dob"]
                if isinstance(dob_str, str):
                    dob_obj = dt.strptime(dob_str, "%Y-%m-%d").date()
                else:
                    dob_obj = dob_str
                age = (dt.now().date() - dob_obj).days // 365
            except Exception as e:
                logger.warning(f"Could not parse DOB: {e}")
                pass
        
        # Validate critical fields before database insert
        name = data.get("name")
        emirates_id = data.get("id_number")
        
        # If critical fields are missing, log warning and use placeholders
        if not name or name == "null":
            logger.warning(f"Emirates ID extraction missing name field. Raw data: {data}")
            name = f"PENDING_EXTRACTION_{application_id[:8]}"
        
        if not emirates_id or emirates_id == "null":
            logger.warning(f"Emirates ID extraction missing ID number")
            emirates_id = f"PENDING_{application_id[:8]}"
        
        # Store in PostgreSQL (structured applicant profile)
        query = """
        INSERT INTO applicants (emirates_id, name, dob, nationality, gender, age)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (emirates_id) DO UPDATE SET
            name = CASE WHEN EXCLUDED.name NOT LIKE 'PENDING%' THEN EXCLUDED.name ELSE applicants.name END,
            dob = COALESCE(EXCLUDED.dob, applicants.dob),
            nationality = COALESCE(EXCLUDED.nationality, applicants.nationality),
            gender = COALESCE(EXCLUDED.gender, applicants.gender),
            age = COALESCE(EXCLUDED.age, applicants.age),
            updated_at = CURRENT_TIMESTAMP
        RETURNING id
        """
        result = await self.postgres.fetchone(
            query,
            emirates_id,
            name,
            dob_obj,  # Use date object
            data.get("nationality"),
            data.get("gender"),
            age,
        )
        
        applicant_id = result["id"] if result else None
        
        # Store in Neo4j (Person node)
        if applicant_id:
            self.neo4j.execute_write(
                """
                MERGE (p:Person {id: $applicant_id})
                SET p.name = $name,
                    p.emirates_id = $emirates_id,
                    p.dob = $dob,
                    p.nationality = $nationality,
                    p.gender = $gender,
                    p.age = $age
                """,
                {
                    "applicant_id": str(applicant_id),
                    "name": data.get("name"),
                    "emirates_id": data.get("id_number"),
                    "dob": data.get("dob"),
                    "nationality": data.get("nationality"),
                    "gender": data.get("gender"),
                    "age": age,
                },
            )
        
        logger.info(f"✓ Stored Emirates ID data (applicant_id: {applicant_id})")
        return str(applicant_id) if applicant_id else None

    async def store_bank_statement(
        self,
        application_id: str,
        applicant_id: str,
        data: Dict[str, Any],
        raw_text: str,
    ):
        """Store bank statement data across databases."""
        # Store in MongoDB (full transactions)
        mongo_doc = {
            "application_id": application_id,
            "document_type": "bank_statement",
            "raw_text": raw_text,
            "structured_data": data,
            "extracted_at": datetime.utcnow().isoformat(),
        }
        self.mongo.insert_one("extractions", mongo_doc)
        
        # Extract key metrics for PostgreSQL
        monthly_income = data.get("total_credits", 0) / 1  # Simplified - assume 1 month statement
        
        # Update application with income
        await self.postgres.execute(
            """
            UPDATE applications
            SET monthly_income = $1
            WHERE id = $2
            """,
            monthly_income,
            application_id,
        )
        
        # Store in Neo4j (Employer node from salary source)
        salary_deposits = data.get("salary_deposits", [])
        if salary_deposits and len(salary_deposits) > 0:
            employer_name = salary_deposits[0].get("source", "Unknown Employer")
            
            # Create Employer node and WORKS_AT relationship
            self.neo4j.execute_write(
                """
                MERGE (e:Employer {name: $employer_name})
                WITH e
                MATCH (p:Person {id: $applicant_id})
                MERGE (p)-[r:WORKS_AT]->(e)
                SET r.source = 'bank_statement',
                    r.monthly_salary = $monthly_salary
                """,
                {
                    "employer_name": employer_name,
                    "applicant_id": applicant_id,
                    "monthly_salary": float(salary_deposits[0].get("amount", 0)),
                },
            )
        
        logger.info(f"✓ Stored bank statement data")

    async def store_resume(
        self,
        application_id: str,
        applicant_id: Optional[str],
        data: Dict[str, Any],
        raw_text: str,
    ):
        """Store resume data across databases."""
        # Store in MongoDB
        mongo_doc = {
            "application_id": application_id,
            "document_type": "resume",
            "raw_text": raw_text,
            "structured_data": data,
            "extracted_at": datetime.utcnow().isoformat(),
        }
        self.mongo.insert_one("extractions", mongo_doc)
        
        # Generate embedding for Qdrant (resume matching)
        resume_text = f"{data.get('professional_summary', '')} {' '.join(data.get('skills', []))}"
        embedding = self.embedder.embed(resume_text)
        
        # Store in Qdrant (only if we have applicant_id)
        if applicant_id is None:
            logger.warning("No applicant_id available, skipping Qdrant storage for resume")
            logger.info("Resume embedding generated but not stored (applicant creation failed)")
            return
        
        # Store in Qdrant
        self.qdrant.upsert_points(
            collection_name="applicant_profiles",
            points=[
                {
                    "id": applicant_id,
                    "vector": embedding if isinstance(embedding, list) else embedding.tolist(),
                    "payload": {
                        "application_id": application_id,
                        "skills": data.get("skills", []),
                        "experience_years": sum(
                            exp.get("duration_months", 0) for exp in data.get("experience", [])
                        ) / 12,
                        "document_type": "resume",
                    },
                }
            ],
        )
        
        # Store in Neo4j (Employer nodes from experience, Skill nodes)
        for exp in data.get("experience", []):
            employer = exp.get("employer", "Unknown")
            title = exp.get("title", "Unknown")
            duration = exp.get("duration_months", 0)
            
            self.neo4j.execute_write(
                """
                MERGE (e:Employer {name: $employer})
                WITH e
                MATCH (p:Person {id: $applicant_id})
                MERGE (p)-[r:WORKED_AT]->(e)
                SET r.title = $title,
                    r.duration_months = $duration,
                    r.source = 'resume'
                """,
                {
                    "employer": employer,
                    "applicant_id": applicant_id,
                    "title": title,
                    "duration": duration,
                },
            )
        
        # Store skills as nodes
        for skill in data.get("skills", [])[:10]:  # Limit to top 10 skills
            self.neo4j.execute_write(
                """
                MERGE (s:Skill {name: $skill})
                WITH s
                MATCH (p:Person {id: $applicant_id})
                MERGE (p)-[:HAS_SKILL]->(s)
                """,
                {"skill": skill, "applicant_id": applicant_id},
            )
        
        logger.info(f"✓ Stored resume data with embedding")

    async def store_assets_liabilities(
        self,
        application_id: str,
        applicant_id: str,
        data: Dict[str, Any],
    ):
        """Store assets/liabilities data."""
        # Store in MongoDB
        mongo_doc = {
            "application_id": application_id,
            "document_type": "assets_liabilities",
            "structured_data": data,
            "extracted_at": datetime.utcnow().isoformat(),
        }
        self.mongo.insert_one("extractions", mongo_doc)
        
        # Calculate net worth for PostgreSQL features
        total_assets = sum(asset.get("value", 0) for asset in data.get("assets", []))
        total_liabilities = sum(liability.get("amount", 0) for liability in data.get("liabilities", []))
        net_worth = total_assets - total_liabilities
        
        # Store in applicant_features table (will be completed by eligibility agent)
        await self.postgres.execute(
            """
            INSERT INTO applicant_features (application_id, total_assets, total_liabilities, net_worth)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (application_id) DO UPDATE SET
                total_assets = EXCLUDED.total_assets,
                total_liabilities = EXCLUDED.total_liabilities,
                net_worth = EXCLUDED.net_worth
            """,
            application_id,
            total_assets,
            total_liabilities,
            net_worth,
        )
        
        logger.info(f"✓ Stored assets/liabilities (net worth: {net_worth})")

    async def store_credit_report(
        self,
        application_id: str,
        applicant_id: str,
        data: Dict[str, Any],
        raw_text: str,
    ):
        """Store credit report data across databases."""
        # Store in MongoDB
        mongo_doc = {
            "application_id": application_id,
            "document_type": "credit_report",
            "raw_text": raw_text,
            "structured_data": data,
            "extracted_at": datetime.utcnow().isoformat(),
        }
        self.mongo.insert_one("extractions", mongo_doc)
        
        # Store credit metrics in features table
        await self.postgres.execute(
            """
            INSERT INTO applicant_features (application_id, credit_score, missed_payments)
            VALUES ($1, $2, $3)
            ON CONFLICT (application_id) DO UPDATE SET
                credit_score = EXCLUDED.credit_score,
                missed_payments = EXCLUDED.missed_payments
            """,
            application_id,
            data.get("credit_score", 0),
            data.get("payment_history", {}).get("missed_payments_count", 0),
        )
        
        # Store Address node in Neo4j (from credit report)
        address = data.get("address", {})
        if address.get("emirate"):
            # Create address hash for uniqueness
            addr_str = f"{address.get('street', '')}{address.get('emirate', '')}"
            addr_hash = hashlib.md5(addr_str.encode()).hexdigest()[:16]
            
            self.neo4j.execute_write(
                """
                MERGE (a:Address {hash: $addr_hash})
                SET a.street = $street,
                    a.emirate = $emirate,
                    a.country = $country
                WITH a
                MATCH (p:Person {id: $applicant_id})
                MERGE (p)-[r:LIVES_AT]->(a)
                SET r.source = 'credit_report'
                """,
                {
                    "addr_hash": addr_hash,
                    "street": address.get("street"),
                    "emirate": address.get("emirate"),
                    "country": address.get("country", "UAE"),
                    "applicant_id": applicant_id,
                },
            )
        
        # Store Employer node from credit report
        employment = data.get("employment", {})
        if employment.get("employer"):
            self.neo4j.execute_write(
                """
                MERGE (e:Employer {name: $employer})
                WITH e
                MATCH (p:Person {id: $applicant_id})
                MERGE (p)-[r:EMPLOYED_AT]->(e)
                SET r.source = 'credit_report',
                    r.position = $position,
                    r.monthly_income = $monthly_income
                """,
                {
                    "employer": employment.get("employer"),
                    "applicant_id": applicant_id,
                    "position": employment.get("position"),
                    "monthly_income": employment.get("monthly_income", 0),
                },
            )
        
        logger.info(f"✓ Stored credit report data")

    async def extract_document(
        self,
        file_path: Path,
        document_type: DocumentType,
        application_id: str,
        applicant_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract and store data from a single document.
        
        ReAct pattern:
        1. Thought: "I need to parse this document"
        2. Action: Call parser
        3. Observation: Raw parser output
        4. Thought: "I need to structure this with LLM"
        5. Action: Call LLM
        6. Observation: Structured JSON
        7. Thought: "I need to store across databases"
        8. Action: Store in PostgreSQL, MongoDB, Neo4j, Qdrant
        9. Observation: Storage confirmation
        
        Args:
            file_path: Path to document
            document_type: Type of document
            application_id: Application ID
            applicant_id: Applicant ID (if already created)
            
        Returns:
            Extraction result dict
        """
        logger.info(f"[ReAct] Extracting {document_type.value}: {file_path.name}")
        
        try:
            # Action: Parse document
            logger.debug(f"[ReAct] Action: Parsing {document_type.value}")
            parse_result = self.parser.parse(file_path, document_type)
            
            if "error" in parse_result.get("data", {}):
                error_msg = parse_result["data"]["error"]
                logger.error(f"[ReAct] Observation: Parser error - {error_msg}")
                return {
                    "document_type": document_type.value,
                    "status": "error",
                    "error": error_msg,
                }
            
            raw_data = parse_result["data"]
            raw_text = parse_result.get("raw_text", "")
            
            logger.debug(f"[ReAct] Observation: Parsed successfully")
            
            # Action: Structure with LLM (if needed)
            if document_type == DocumentType.ASSETS_LIABILITIES:
                # Excel data is already structured
                structured_data = raw_data
            elif document_type == DocumentType.EMIRATES_ID:
                # OCR already extracts structured data, use it directly
                logger.debug(f"[ReAct] Using OCR structured data (skipping LLM)")
                structured_data = raw_data  # raw_data from OCR is already structured
            else:
                logger.debug(f"[ReAct] Action: Structuring with LLM")
                structured_data = self.structure_with_llm(raw_text, document_type, trace_id)
                logger.debug(f"[ReAct] Observation: LLM structured data")
            
            # Action: Store across databases
            logger.debug(f"[ReAct] Action: Storing across databases")
            
            if document_type == DocumentType.EMIRATES_ID:
                applicant_id = await self.store_emirates_id(
                    application_id, structured_data, raw_text
                )
            elif document_type == DocumentType.BANK_STATEMENT:
                await self.store_bank_statement(
                    application_id, applicant_id, structured_data, raw_text
                )
            elif document_type == DocumentType.RESUME:
                await self.store_resume(
                    application_id, applicant_id, structured_data, raw_text
                )
            elif document_type == DocumentType.ASSETS_LIABILITIES:
                await self.store_assets_liabilities(
                    application_id, applicant_id, structured_data
                )
            elif document_type == DocumentType.CREDIT_REPORT:
                await self.store_credit_report(
                    application_id, applicant_id, structured_data, raw_text
                )
            
            logger.info(f"[ReAct] ✓ Extraction complete: {document_type.value}")
            
            return {
                "document_type": document_type.value,
                "status": "success",
                "data": structured_data,
                "applicant_id": applicant_id,
            }
            
        except Exception as e:
            logger.error(f"[ReAct] Extraction failed for {document_type.value}: {e}", exc_info=True)
            return {
                "document_type": document_type.value,
                "status": "error",
                "error": str(e),
            }


async def extract_data_agent(state: ApplicationState) -> ApplicationState:
    """Data Extraction Agent node for LangGraph.
    
    Processes all uploaded documents through ReAct loop:
    - Parse → Structure with LLM → Store across databases
    
    Args:
        state: Current application state
        
    Returns:
        Updated state with extraction_results
    """
    logger.info("=" * 80)
    logger.info("EXTRACTION AGENT STARTED")
    logger.info("=" * 80)
    
    state["current_agent"] = "extraction"
    state["processing_status"] = "extracting"
    
    # Start tracing
    trace_id = trace_agent_start(
        application_id=state["application_id"],
        agent_name="extraction",
        input_state=state,
    )
    
    agent = ExtractionAgent()
    
    try:
        await agent.connect_databases()
        
        uploaded_docs = state.get("uploaded_documents", [])
        logger.info(f"Processing {len(uploaded_docs)} documents")
        
        extraction_results = {}
        applicant_id = None
        
        # Process Emirates ID first to create applicant record
        emirates_id_doc = next(
            (d for d in uploaded_docs if "emirates" in d["file_name"].lower() or "id" in d["file_name"].lower()),
            None
        )
        
        if emirates_id_doc:
            logger.info("Processing Emirates ID first to create applicant record")
            result = await agent.extract_document(
                Path(emirates_id_doc["file_path"]),
                DocumentType.EMIRATES_ID,
                state["application_id"],
                trace_id=trace_id,
            )
            extraction_results[DocumentType.EMIRATES_ID.value] = result
            applicant_id = result.get("applicant_id")
            state["applicant_id"] = applicant_id
        
        # Process remaining documents
        for doc in uploaded_docs:
            file_path = Path(doc["file_path"])
            
            # Skip if already processed
            if "emirates" in file_path.name.lower() or "_id" in file_path.name.lower():
                continue
            
            # Detect document type
            doc_type = agent.parser.detect_document_type(file_path)
            
            if doc_type == DocumentType.UNKNOWN:
                logger.warning(f"Unknown document type: {file_path.name}")
                continue
            
            result = await agent.extract_document(
                file_path,
                doc_type,
                state["application_id"],
                applicant_id,
                trace_id=trace_id,
            )
            extraction_results[doc_type.value] = result
        
        # Update state
        state["extraction_results"] = extraction_results
        state["extraction_complete"] = True
        
        # Count successes/failures
        success_count = sum(1 for r in extraction_results.values() if r.get("status") == "success")
        error_count = sum(1 for r in extraction_results.values() if r.get("status") == "error")
        
        logger.info("=" * 80)
        logger.info(f"EXTRACTION AGENT COMPLETE: {success_count} succeeded, {error_count} errors")
        logger.info("=" * 80)
        
        # End tracing
        trace_agent_end(
            trace_id=trace_id,
            agent_name="extraction",
            output_state=state,
            success=(error_count == 0),
        )
        
    except Exception as e:
        logger.error(f"Extraction agent failed: {e}", exc_info=True)
        state["error"] = f"Extraction failed: {str(e)}"
        state["processing_status"] = "error"
        
        # End tracing with error
        trace_agent_end(
            trace_id=trace_id,
            agent_name="extraction",
            output_state=state,
            success=False,
        )
    
    finally:
        await agent.disconnect_databases()
    
    return state


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    # Test with a sample document
    from src.agents.state import create_initial_state
    
    async def test():
        state = create_initial_state("test-app-001", "APP-001")
        state["uploaded_documents"] = [
            {
                "file_path": "data/synthetic/output/applicant_001/emirates_id.png",
                "file_name": "emirates_id.png",
            }
        ]
        
        result_state = await extract_data_agent(state)
        print(f"\nExtraction complete: {result_state['extraction_complete']}")
        print(f"Results: {list(result_state['extraction_results'].keys())}")
    
    asyncio.run(test())
