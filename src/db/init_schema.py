"""Initialize all database schemas."""

import asyncio
import logging
import sys

from src.db.postgres import initialize_postgres_schema
from src.db.mongo import initialize_mongo_collections
from src.db.qdrant_client import initialize_qdrant_collections
from src.db.neo4j_client import initialize_neo4j_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def initialize_all_databases():
    """Initialize all database schemas and collections."""
    try:
        logger.info("=" * 60)
        logger.info("Starting database initialization...")
        logger.info("=" * 60)
        
        # PostgreSQL (async)
        logger.info("\n[1/4] Initializing PostgreSQL...")
        await initialize_postgres_schema()
        
        # MongoDB (sync)
        logger.info("\n[2/4] Initializing MongoDB...")
        initialize_mongo_collections()
        
        # Qdrant (sync)
        logger.info("\n[3/4] Initializing Qdrant...")
        initialize_qdrant_collections()
        
        # Neo4j (sync)
        logger.info("\n[4/4] Initializing Neo4j...")
        initialize_neo4j_schema()
        
        logger.info("=" * 60)
        logger.info("All databases initialized successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(initialize_all_databases())
