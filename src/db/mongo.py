"""MongoDB client and collection initialization."""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

import pymongo
from pymongo import MongoClient as PyMongoClient
from pymongo.database import Database
from pymongo.collection import Collection

from src.config import settings

logger = logging.getLogger(__name__)


class MongoClient:
    """MongoDB database client."""

    def __init__(self):
        self.client: Optional[PyMongoClient] = None
        self.db: Optional[Database] = None

    def connect(self):
        """Connect to MongoDB."""
        if self.client is None:
            logger.info(f"Connecting to MongoDB at {settings.mongo_host}:{settings.mongo_port}")
            self.client = PyMongoClient(settings.mongo_url)
            self.db = self.client[settings.mongo_db]
            logger.info(f"Connected to MongoDB database: {settings.mongo_db}")

    def disconnect(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            logger.info("MongoDB connection closed")

    def get_collection(self, name: str) -> Collection:
        """Get a collection by name."""
        if self.db is None:
            raise RuntimeError("Not connected to MongoDB. Call connect() first.")
        return self.db[name]

    def insert_one(self, collection: str, document: Dict[str, Any]) -> str:
        """Insert a single document."""
        result = self.get_collection(collection).insert_one(document)
        return str(result.inserted_id)

    def insert_many(self, collection: str, documents: List[Dict[str, Any]]) -> List[str]:
        """Insert multiple documents."""
        result = self.get_collection(collection).insert_many(documents)
        return [str(oid) for oid in result.inserted_ids]

    def find_one(self, collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single document."""
        return self.get_collection(collection).find_one(query)

    def find(self, collection: str, query: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Find multiple documents."""
        return list(self.get_collection(collection).find(query).limit(limit))

    def update_one(self, collection: str, query: Dict[str, Any], update: Dict[str, Any]) -> bool:
        """Update a single document."""
        result = self.get_collection(collection).update_one(query, update)
        return result.modified_count > 0

    def delete_one(self, collection: str, query: Dict[str, Any]) -> bool:
        """Delete a single document."""
        result = self.get_collection(collection).delete_one(query)
        return result.deleted_count > 0


def initialize_mongo_collections():
    """Initialize MongoDB collections with indexes."""
    client = MongoClient()
    try:
        client.connect()
        
        logger.info("Initializing MongoDB collections...")
        
        # raw_documents collection
        raw_docs = client.get_collection("raw_documents")
        raw_docs.create_index([("application_id", pymongo.ASCENDING)])
        raw_docs.create_index([("document_type", pymongo.ASCENDING)])
        raw_docs.create_index([("uploaded_at", pymongo.DESCENDING)])
        logger.info("Created collection: raw_documents")
        
        # extractions collection
        extractions = client.get_collection("extractions")
        extractions.create_index([("application_id", pymongo.ASCENDING)])
        extractions.create_index([("document_type", pymongo.ASCENDING)])
        extractions.create_index([("extracted_at", pymongo.DESCENDING)])
        extractions.create_index([("extraction_version", pymongo.ASCENDING)])
        logger.info("Created collection: extractions")
        
        logger.info("MongoDB collections initialized successfully")
        
        # List all collections
        collections = client.db.list_collection_names()
        logger.info(f"Available collections: {collections}")
        
    finally:
        client.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initialize_mongo_collections()
