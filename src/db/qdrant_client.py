"""Qdrant vector database client."""

import logging
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient as QdrantSDK
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.config import settings

logger = logging.getLogger(__name__)


class QdrantClientWrapper:
    """Qdrant vector database client."""

    def __init__(self):
        self.client: Optional[QdrantSDK] = None
        self.collection_name = "applicant_profiles"
        self.vector_size = 384  # all-MiniLM-L6-v2 embedding dimension

    def connect(self):
        """Connect to Qdrant."""
        if self.client is None:
            logger.info(f"Connecting to Qdrant at {settings.qdrant_url}")
            self.client = QdrantSDK(url=settings.qdrant_url)
            logger.info("Connected to Qdrant")

    def disconnect(self):
        """Close Qdrant connection."""
        if self.client:
            self.client.close()
            self.client = None
            logger.info("Qdrant connection closed")

    def create_collection(self, collection_name: str, vector_size: int = 384):
        """Create a collection if it doesn't exist."""
        if self.client is None:
            raise RuntimeError("Not connected to Qdrant. Call connect() first.")
        
        collections = [c.name for c in self.client.get_collections().collections]
        
        if collection_name not in collections:
            logger.info(f"Creating Qdrant collection: {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Created collection: {collection_name}")
        else:
            logger.info(f"Collection already exists: {collection_name}")

    def upsert_points(
        self,
        collection_name: str,
        points: List[Dict[str, Any]],
    ):
        """Upsert points (vectors with payloads) into a collection.
        
        Args:
            collection_name: Name of the collection
            points: List of dicts with 'id', 'vector', and 'payload' keys
        """
        if self.client is None:
            raise RuntimeError("Not connected to Qdrant. Call connect() first.")
        
        qdrant_points = [
            PointStruct(
                id=point["id"],
                vector=point["vector"],
                payload=point.get("payload", {}),
            )
            for point in points
        ]
        
        self.client.upsert(
            collection_name=collection_name,
            points=qdrant_points,
        )
        logger.info(f"Upserted {len(points)} points to collection: {collection_name}")

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            limit: Number of results to return
            score_threshold: Minimum score threshold
            filter_dict: Optional payload filter
            
        Returns:
            List of search results with id, score, and payload
        """
        if self.client is None:
            raise RuntimeError("Not connected to Qdrant. Call connect() first.")
        
        query_filter = None
        if filter_dict:
            # Simple filter construction - extend as needed
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            query_filter = Filter(must=conditions)
        
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results
        ]


def initialize_qdrant_collections():
    """Initialize Qdrant collections."""
    client = QdrantClientWrapper()
    try:
        client.connect()
        
        logger.info("Initializing Qdrant collections...")
        
        # Create main collections
        client.create_collection("applicant_profiles", vector_size=384)
        client.create_collection("enablement_programs", vector_size=384)
        client.create_collection("policy_documents", vector_size=384)
        
        logger.info("Qdrant collections initialized successfully")
        
        # List all collections
        collections = client.client.get_collections().collections
        logger.info(f"Available collections: {[c.name for c in collections]}")
        
    finally:
        client.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initialize_qdrant_collections()
