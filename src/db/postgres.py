"""PostgreSQL client and schema initialization."""

import asyncpg
import logging
from typing import Any, Dict, List, Optional

from src.config import settings

logger = logging.getLogger(__name__)


class PostgresClient:
    """PostgreSQL database client with connection pooling."""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Create connection pool."""
        if self.pool is None:
            logger.info(f"Connecting to PostgreSQL at {settings.postgres_host}:{settings.postgres_port}")
            self.pool = await asyncpg.create_pool(
                host=settings.postgres_host,
                port=settings.postgres_port,
                user=settings.postgres_user,
                password=settings.postgres_password,
                database=settings.postgres_db,
                min_size=2,
                max_size=10,
                command_timeout=60,
                ssl=False,  # Disable SSL for local development
            )
            logger.info("PostgreSQL connection pool created")

    async def disconnect(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("PostgreSQL connection pool closed")

    async def execute(self, query: str, *args) -> str:
        """Execute a query that doesn't return rows."""
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        """Fetch multiple rows as dictionaries."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]

    async def fetchone(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Fetch a single row as a dictionary."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    async def fetchval(self, query: str, *args) -> Any:
        """Fetch a single value."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)


# Schema initialization SQL
SCHEMA_SQL = """
-- ──── Applicants table ────
CREATE TABLE IF NOT EXISTS applicants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    emirates_id VARCHAR(50) UNIQUE NOT NULL,
    dob DATE,  -- Nullable: will be filled during extraction from Emirates ID
    nationality VARCHAR(100) DEFAULT 'Unknown',
    gender VARCHAR(10),
    age INTEGER,
    emirate VARCHAR(100),
    contact_email VARCHAR(255),
    contact_phone VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_applicants_emirates_id ON applicants(emirates_id);
CREATE INDEX IF NOT EXISTS idx_applicants_created_at ON applicants(created_at);

-- ──── Applications table ────
CREATE TABLE IF NOT EXISTS applications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    applicant_id UUID NOT NULL REFERENCES applicants(id) ON DELETE CASCADE,
    application_ref VARCHAR(50) UNIQUE NOT NULL,
    status VARCHAR(50) DEFAULT 'processing',
    household_size INTEGER,
    num_dependents INTEGER,
    monthly_income DECIMAL(12, 2),
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_applications_applicant_id ON applications(applicant_id);
CREATE INDEX IF NOT EXISTS idx_applications_ref ON applications(application_ref);
CREATE INDEX IF NOT EXISTS idx_applications_status ON applications(status);

-- ──── Extraction results table ────
CREATE TABLE IF NOT EXISTS extraction_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    application_id UUID NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
    document_type VARCHAR(100) NOT NULL,
    extraction_data JSONB NOT NULL,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_extraction_application_id ON extraction_results(application_id);
CREATE INDEX IF NOT EXISTS idx_extraction_document_type ON extraction_results(document_type);

-- ──── Validation results table ────
CREATE TABLE IF NOT EXISTS validation_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    application_id UUID NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
    issue_type VARCHAR(100) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    source_doc_a VARCHAR(100),
    source_doc_b VARCHAR(100),
    field_name VARCHAR(100),
    value_a TEXT,
    value_b TEXT,
    description TEXT,
    validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_validation_application_id ON validation_results(application_id);
CREATE INDEX IF NOT EXISTS idx_validation_severity ON validation_results(severity);

-- ──── Applicant features table (for classifier) ────
CREATE TABLE IF NOT EXISTS applicant_features (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    application_id UUID NOT NULL UNIQUE REFERENCES applications(id) ON DELETE CASCADE,
    monthly_income DECIMAL(12, 2),
    employment_months INTEGER,
    employment_status VARCHAR(50),
    household_size INTEGER,
    num_dependents INTEGER,
    income_per_family DECIMAL(12, 2),
    total_assets DECIMAL(12, 2),
    total_liabilities DECIMAL(12, 2),
    net_worth DECIMAL(12, 2),
    debt_to_income DECIMAL(8, 4),
    credit_score INTEGER,
    missed_payments INTEGER,
    age INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_features_application_id ON applicant_features(application_id);

-- ──── Eligibility scores table ────
CREATE TABLE IF NOT EXISTS eligibility_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    application_id UUID NOT NULL UNIQUE REFERENCES applications(id) ON DELETE CASCADE,
    score DECIMAL(5, 4) NOT NULL,
    category VARCHAR(50) NOT NULL,
    feature_importances JSONB,
    notes TEXT,
    scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_eligibility_application_id ON eligibility_scores(application_id);
CREATE INDEX IF NOT EXISTS idx_eligibility_category ON eligibility_scores(category);

-- ──── Decisions table (audit trail) ────
CREATE TABLE IF NOT EXISTS decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    application_id UUID NOT NULL UNIQUE REFERENCES applications(id) ON DELETE CASCADE,
    decision VARCHAR(50) NOT NULL,
    confidence DECIMAL(5, 4),
    reasoning TEXT NOT NULL,
    key_factors JSONB,
    enablement_programs JSONB,
    review_notes TEXT,
    decided_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_decisions_application_id ON decisions(application_id);
CREATE INDEX IF NOT EXISTS idx_decisions_decision ON decisions(decision);

-- ──── Audit log table ────
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    application_id UUID REFERENCES applications(id) ON DELETE SET NULL,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_application_id ON audit_log(application_id);
CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_log(created_at);
"""


async def initialize_postgres_schema():
    """Initialize PostgreSQL schema."""
    client = PostgresClient()
    try:
        await client.connect()
        
        logger.info("Initializing PostgreSQL schema...")
        
        # Execute schema creation
        async with client.pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)
        
        logger.info("PostgreSQL schema initialized successfully")
        
        # Verify tables created
        tables = await client.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """)
        
        logger.info(f"Created tables: {[t['table_name'] for t in tables]}")
        
    finally:
        await client.disconnect()


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(initialize_postgres_schema())
