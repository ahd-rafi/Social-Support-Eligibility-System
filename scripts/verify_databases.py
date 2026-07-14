"""Verify database connectivity and check record counts."""

import asyncio
import logging
import sys
from typing import Dict, Any

from src.db.postgres import PostgresClient
from src.db.mongo import MongoClient
from src.db.qdrant_client import QdrantClientWrapper
from src.db.neo4j_client import Neo4jClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def verify_postgres() -> Dict[str, Any]:
    """Verify PostgreSQL connectivity and get table counts."""
    client = PostgresClient()
    try:
        await client.connect()
        logger.info("✓ PostgreSQL connection successful")
        
        # Get table counts
        tables = await client.fetch("""
            SELECT 
                table_name,
                (xpath('/row/c/text()', query_to_xml(format('SELECT COUNT(*) AS c FROM %I', table_name), false, true, '')))[1]::text::int AS row_count
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        
        counts = {t['table_name']: t['row_count'] for t in tables}
        
        logger.info(f"  Tables found: {len(counts)}")
        for table, count in counts.items():
            logger.info(f"    - {table}: {count} rows")
        
        return {"status": "ok", "tables": counts}
        
    except Exception as e:
        logger.error(f"✗ PostgreSQL connection failed: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        await client.disconnect()


def verify_mongo() -> Dict[str, Any]:
    """Verify MongoDB connectivity and get collection counts."""
    client = MongoClient()
    try:
        client.connect()
        logger.info("✓ MongoDB connection successful")
        
        # Get collection counts
        collections = client.db.list_collection_names()
        counts = {}
        for coll in collections:
            counts[coll] = client.get_collection(coll).count_documents({})
        
        logger.info(f"  Collections found: {len(counts)}")
        for coll, count in counts.items():
            logger.info(f"    - {coll}: {count} documents")
        
        return {"status": "ok", "collections": counts}
        
    except Exception as e:
        logger.error(f"✗ MongoDB connection failed: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        client.disconnect()


def verify_qdrant() -> Dict[str, Any]:
    """Verify Qdrant connectivity and get collection info."""
    client = QdrantClientWrapper()
    try:
        client.connect()
        logger.info("✓ Qdrant connection successful")
        
        # Get collections
        collections = client.client.get_collections().collections
        
        collection_info = {}
        for coll in collections:
            info = client.client.get_collection(coll.name)
            collection_info[coll.name] = {
                "points_count": info.points_count,
                "vector_size": info.config.params.vectors.size,
            }
        
        logger.info(f"  Collections found: {len(collection_info)}")
        for name, info in collection_info.items():
            logger.info(f"    - {name}: {info['points_count']} points ({info['vector_size']}-dim)")
        
        return {"status": "ok", "collections": collection_info}
        
    except Exception as e:
        logger.error(f"✗ Qdrant connection failed: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        client.disconnect()


def verify_neo4j() -> Dict[str, Any]:
    """Verify Neo4j connectivity and get node/relationship counts."""
    client = Neo4jClient()
    try:
        client.connect()
        logger.info("✓ Neo4j connection successful")
        
        # Get node counts by label
        node_result = client.execute_query("""
            CALL db.labels() YIELD label
            CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {}) YIELD value
            RETURN label, value.count AS count
            ORDER BY label
        """)
        
        # Fallback if APOC is not available
        if not node_result:
            node_result = []
            labels_result = client.execute_query("CALL db.labels()")
            for row in labels_result:
                label = row['label']
                count_result = client.execute_query(f"MATCH (n:{label}) RETURN count(n) as count")
                node_result.append({"label": label, "count": count_result[0]['count']})
        
        # Get relationship counts by type
        rel_result = client.execute_query("""
            CALL db.relationshipTypes() YIELD relationshipType
            CALL apoc.cypher.run('MATCH ()-[r:' + relationshipType + ']->() RETURN count(r) as count', {}) YIELD value
            RETURN relationshipType, value.count AS count
            ORDER BY relationshipType
        """)
        
        # Fallback if APOC is not available
        if not rel_result:
            rel_result = []
            types_result = client.execute_query("CALL db.relationshipTypes()")
            for row in types_result:
                rel_type = row['relationshipType']
                count_result = client.execute_query(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                rel_result.append({"relationshipType": rel_type, "count": count_result[0]['count']})
        
        nodes = {r['label']: r['count'] for r in node_result}
        relationships = {r['relationshipType']: r['count'] for r in rel_result}
        
        logger.info(f"  Node labels: {len(nodes)}")
        for label, count in nodes.items():
            logger.info(f"    - {label}: {count} nodes")
        
        logger.info(f"  Relationship types: {len(relationships)}")
        for rel_type, count in relationships.items():
            logger.info(f"    - {rel_type}: {count} relationships")
        
        return {"status": "ok", "nodes": nodes, "relationships": relationships}
        
    except Exception as e:
        logger.error(f"✗ Neo4j connection failed: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        client.disconnect()


async def main():
    """Run all database verifications."""
    logger.info("=" * 60)
    logger.info("Database Connectivity Verification")
    logger.info("=" * 60)
    
    results = {}
    
    logger.info("\n[1/4] Verifying PostgreSQL...")
    results['postgres'] = await verify_postgres()
    
    logger.info("\n[2/4] Verifying MongoDB...")
    results['mongo'] = verify_mongo()
    
    logger.info("\n[3/4] Verifying Qdrant...")
    results['qdrant'] = verify_qdrant()
    
    logger.info("\n[4/4] Verifying Neo4j...")
    results['neo4j'] = verify_neo4j()
    
    logger.info("\n" + "=" * 60)
    
    # Summary
    all_ok = all(r['status'] == 'ok' for r in results.values())
    
    if all_ok:
        logger.info("✓ All databases are accessible and healthy!")
    else:
        logger.error("✗ Some databases failed connectivity checks")
        for db, result in results.items():
            if result['status'] != 'ok':
                logger.error(f"  - {db}: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
