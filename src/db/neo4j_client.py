"""Neo4j graph database client."""

import logging
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Driver, Session

from src.config import settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j graph database client."""

    def __init__(self):
        self.driver: Optional[Driver] = None

    def connect(self):
        """Connect to Neo4j."""
        if self.driver is None:
            logger.info(f"Connecting to Neo4j at {settings.neo4j_uri}")
            self.driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j")

    def disconnect(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            self.driver = None
            logger.info("Neo4j connection closed")

    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Target database (defaults to neo4j)
            
        Returns:
            List of result records as dictionaries
        """
        if self.driver is None:
            raise RuntimeError("Not connected to Neo4j. Call connect() first.")
        
        with self.driver.session(database=database or settings.neo4j_database) as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]

    def execute_write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a write transaction."""
        if self.driver is None:
            raise RuntimeError("Not connected to Neo4j. Call connect() first.")
        
        def _execute(tx):
            result = tx.run(query, parameters or {})
            return [dict(record) for record in result]
        
        with self.driver.session(database=database or settings.neo4j_database) as session:
            return session.execute_write(_execute)

    def create_node(
        self,
        label: str,
        properties: Dict[str, Any],
        database: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a node with given label and properties."""
        query = f"""
        CREATE (n:{label} $props)
        RETURN n
        """
        result = self.execute_write(query, {"props": properties}, database)
        return result[0]["n"] if result else {}

    def create_relationship(
        self,
        from_label: str,
        from_property: str,
        from_value: Any,
        to_label: str,
        to_property: str,
        to_value: Any,
        rel_type: str,
        rel_properties: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ):
        """Create a relationship between two nodes."""
        query = f"""
        MATCH (a:{from_label} {{{from_property}: $from_value}})
        MATCH (b:{to_label} {{{to_property}: $to_value}})
        CREATE (a)-[r:{rel_type} $rel_props]->(b)
        RETURN r
        """
        return self.execute_write(
            query,
            {
                "from_value": from_value,
                "to_value": to_value,
                "rel_props": rel_properties or {},
            },
            database,
        )


def initialize_neo4j_schema():
    """Initialize Neo4j constraints and indexes."""
    client = Neo4jClient()
    try:
        client.connect()
        
        logger.info("Initializing Neo4j schema...")
        
        # Create constraints (unique identifiers)
        constraints = [
            "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT applicant_id IF NOT EXISTS FOR (a:Applicant) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT employer_name IF NOT EXISTS FOR (e:Employer) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT address_hash IF NOT EXISTS FOR (a:Address) REQUIRE a.hash IS UNIQUE",
        ]
        
        for constraint in constraints:
            try:
                client.execute_write(constraint)
                logger.info(f"Created constraint: {constraint.split()[2]}")
            except Exception as e:
                if "already exists" in str(e):
                    logger.info(f"Constraint already exists: {constraint.split()[2]}")
                else:
                    raise
        
        # Create indexes for common queries
        indexes = [
            "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)",
            "CREATE INDEX applicant_emirates_id IF NOT EXISTS FOR (a:Applicant) ON (a.emirates_id)",
            "CREATE INDEX employer_name_idx IF NOT EXISTS FOR (e:Employer) ON (e.name)",
            "CREATE INDEX address_emirate IF NOT EXISTS FOR (a:Address) ON (a.emirate)",
        ]
        
        for index in indexes:
            try:
                client.execute_write(index)
                logger.info(f"Created index: {index.split()[2]}")
            except Exception as e:
                if "already exists" in str(e):
                    logger.info(f"Index already exists: {index.split()[2]}")
                else:
                    raise
        
        logger.info("Neo4j schema initialized successfully")
        
        # Verify schema
        result = client.execute_query("SHOW CONSTRAINTS")
        logger.info(f"Active constraints: {len(result)}")
        
        result = client.execute_query("SHOW INDEXES")
        logger.info(f"Active indexes: {len(result)}")
        
    finally:
        client.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initialize_neo4j_schema()
