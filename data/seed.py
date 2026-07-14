"""Seed all databases with synthetic data."""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import hashlib

from src.db.postgres import PostgresClient
from src.db.mongo import MongoClient
from src.db.qdrant_client import QdrantClientWrapper
from src.db.neo4j_client import Neo4jClient
from src.ml.embeddings import get_embedding_generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYNTHETIC_DATA_DIR = Path(__file__).parent / "synthetic" / "output"
PROGRAMS_FILE = Path(__file__).parent / "programs" / "programs.json"


class DatabaseSeeder:
    """Seed all four databases with synthetic data."""

    def __init__(self):
        self.pg_client = PostgresClient()
        self.mongo_client = MongoClient()
        self.qdrant_client = QdrantClientWrapper()
        self.neo4j_client = Neo4jClient()
        self.embedding_gen = None

    async def connect_all(self):
        """Connect to all databases."""
        logger.info("Connecting to all databases...")
        await self.pg_client.connect()
        self.mongo_client.connect()
        self.qdrant_client.connect()
        self.neo4j_client.connect()
        logger.info("✓ All databases connected")

    async def disconnect_all(self):
        """Disconnect from all databases."""
        logger.info("Disconnecting from all databases...")
        await self.pg_client.disconnect()
        self.mongo_client.disconnect()
        self.qdrant_client.disconnect()
        self.neo4j_client.disconnect()
        logger.info("✓ All databases disconnected")

    def load_profiles(self) -> List[Dict[str, Any]]:
        """Load all synthetic profiles."""
        profiles = []
        
        if not SYNTHETIC_DATA_DIR.exists():
            raise FileNotFoundError(
                f"Synthetic data directory not found: {SYNTHETIC_DATA_DIR}\n"
                "Run: python -m data.synthetic.generate first"
            )
        
        for profile_dir in SYNTHETIC_DATA_DIR.iterdir():
            if profile_dir.is_dir() and profile_dir.name.startswith("applicant_"):
                profile_file = profile_dir / "profile.json"
                if profile_file.exists():
                    with open(profile_file) as f:
                        profile = json.load(f)
                        profile["directory"] = profile_dir
                        profiles.append(profile)
        
        logger.info(f"Loaded {len(profiles)} synthetic profiles")
        return sorted(profiles, key=lambda p: p["profile_id"])

    def load_programs(self) -> List[Dict[str, Any]]:
        """Load enablement programs."""
        with open(PROGRAMS_FILE) as f:
            programs = json.load(f)
        logger.info(f"Loaded {len(programs)} enablement programs")
        return programs

    async def seed_postgres(self, profiles: List[Dict[str, Any]]):
        """Seed PostgreSQL with applicant data."""
        from datetime import date
        
        logger.info("\n[1/4] Seeding PostgreSQL...")
        
        # Clear existing data first (CASCADE will delete related records)
        logger.info("  Clearing existing data...")
        await self.pg_client.execute("TRUNCATE TABLE applicants CASCADE")
        logger.info("  ✓ Cleared existing data")
        
        for profile in profiles:
            # Convert dob string to date object
            dob = date.fromisoformat(profile["dob"]) if isinstance(profile["dob"], str) else profile["dob"]
            
            # Insert applicant
            applicant_id = await self.pg_client.fetchval("""
                INSERT INTO applicants (
                    name, emirates_id, dob, nationality, gender, age, emirate,
                    contact_email, contact_phone
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
            """,
                profile["name"],
                profile["emirates_id"],
                dob,
                profile["nationality"],
                profile["gender"],
                profile["age"],
                profile["emirate"],
                profile["email"],
                profile["phone"],
            )
            
            # Insert application
            app_ref = f"APP-{profile['profile_id'].split('_')[1]}"
            application_id = await self.pg_client.fetchval("""
                INSERT INTO applications (
                    applicant_id, application_ref, status, household_size,
                    num_dependents, monthly_income
                ) VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
            """,
                applicant_id,
                app_ref,
                "processing",
                profile["household_size"],
                profile["num_dependents"],
                profile["monthly_income"],
            )
            
            # Insert features (for classifier training later)
            await self.pg_client.execute("""
                INSERT INTO applicant_features (
                    application_id, monthly_income, employment_months, employment_status,
                    household_size, num_dependents, income_per_family,
                    total_assets, total_liabilities, net_worth, debt_to_income,
                    credit_score, missed_payments, age
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """,
                application_id,
                profile["monthly_income"],
                profile["employment_months"],
                profile["employment_status"],
                profile["household_size"],
                profile["num_dependents"],
                profile["income_per_family"],
                profile["total_assets"],
                profile["total_liabilities"],
                profile["net_worth"],
                profile["debt_to_income"],
                profile["credit_score"],
                profile["missed_payments"],
                profile["age"],
            )
            
            # Store IDs for later use
            profile["_applicant_id"] = str(applicant_id)
            profile["_application_id"] = str(application_id)
            
            logger.debug(f"  ✓ Seeded {profile['name']}")
        
        logger.info(f"✓ PostgreSQL seeded with {len(profiles)} profiles")

    def seed_mongo(self, profiles: List[Dict[str, Any]]):
        """Seed MongoDB with raw document metadata."""
        logger.info("\n[2/4] Seeding MongoDB...")
        
        # Clear existing data
        logger.info("  Clearing existing data...")
        self.mongo_client.db["raw_documents"].delete_many({})
        logger.info("  ✓ Cleared existing data")
        
        for profile in profiles:
            # Store raw document metadata
            doc_meta = {
                "application_id": profile["_application_id"],
                "profile_id": profile["profile_id"],
                "documents": {
                    "emirates_id": {
                        "filename": "emirates_id.png",
                        "path": str(profile["directory"] / "emirates_id.png"),
                        "type": "image/png",
                    }
                },
                "uploaded_at": "2024-01-01T00:00:00Z",  # Placeholder
            }
            
            self.mongo_client.insert_one("raw_documents", doc_meta)
            
            # Store extraction placeholder (will be filled by agents)
            extraction_doc = {
                "application_id": profile["_application_id"],
                "profile_id": profile["profile_id"],
                "document_type": "emirates_id",
                "extraction_data": {
                    "name": profile["name"],
                    "emirates_id": profile["emirates_id"],
                    "dob": profile["dob"],
                    "nationality": profile["nationality"],
                    "gender": profile["gender"],
                    "emirate": profile["emirate"],
                },
                "extraction_version": 1,
                "extracted_at": "2024-01-01T00:00:00Z",
            }
            
            self.mongo_client.insert_one("extractions", extraction_doc)
            
            logger.debug(f"  ✓ Seeded MongoDB for {profile['name']}")
        
        logger.info(f"✓ MongoDB seeded with {len(profiles)} documents")

    def seed_qdrant(self, profiles: List[Dict[str, Any]], programs: List[Dict[str, Any]]):
        """Seed Qdrant with embeddings."""
        logger.info("\n[3/4] Seeding Qdrant...")
        
        # Clear existing collections (recreate them)
        logger.info("  Clearing existing data...")
        try:
            self.qdrant_client.client.delete_collection("applicant_profiles")
        except:
            pass
        try:
            self.qdrant_client.client.delete_collection("enablement_programs")
        except:
            pass
        
        # Recreate collections
        self.qdrant_client.create_collection(
            "applicant_profiles",
            vector_size=384
        )
        self.qdrant_client.create_collection(
            "enablement_programs",
            vector_size=384
        )
        logger.info("  ✓ Cleared existing data")
        
        # Initialize embedding generator
        if self.embedding_gen is None:
            self.embedding_gen = get_embedding_generator()
        
        # Generate resume embeddings
        logger.info("  Generating resume embeddings...")
        resume_texts = []
        for profile in profiles:
            # Create resume summary text
            resume_text = (
                f"{profile['name']} has {profile['employment_months']} months of experience "
                f"in {profile['sector']} sector"
            )
            if profile["employer"]:
                resume_text += f" working at {profile['employer']}"
            resume_text += f". Skills: {', '.join(profile['skills'])}."
            resume_texts.append(resume_text)
        
        resume_embeddings = self.embedding_gen.embed_batch(resume_texts)
        
        # Upsert to Qdrant
        points = []
        for i, profile in enumerate(profiles):
            points.append({
                "id": i + 1,
                "vector": resume_embeddings[i],
                "payload": {
                    "profile_id": profile["profile_id"],
                    "application_id": profile["_application_id"],
                    "name": profile["name"],
                    "sector": profile["sector"],
                    "skills": profile["skills"],
                    "experience_months": profile["employment_months"],
                },
            })
        
        self.qdrant_client.upsert_points("applicant_profiles", points)
        logger.info(f"  ✓ Upserted {len(points)} resume embeddings")
        
        # Generate program embeddings
        logger.info("  Generating program embeddings...")
        program_texts = [
            f"{p['name']}. {p['description']}" for p in programs
        ]
        program_embeddings = self.embedding_gen.embed_batch(program_texts)
        
        points = []
        for i, program in enumerate(programs):
            points.append({
                "id": i + 1,
                "vector": program_embeddings[i],
                "payload": {
                    "program_id": program["id"],
                    "name": program["name"],
                    "category": program["category"],
                    "sector": program["sector"],
                    "description": program["description"],
                    "duration_months": program["duration_months"],
                    "provider": program["provider"],
                },
            })
        
        self.qdrant_client.upsert_points("enablement_programs", points)
        logger.info(f"  ✓ Upserted {len(points)} program embeddings")
        
        logger.info(f"✓ Qdrant seeded with {len(profiles)} profiles + {len(programs)} programs")

    def seed_neo4j(self, profiles: List[Dict[str, Any]]):
        """Seed Neo4j with entity graph."""
        logger.info("\n[4/4] Seeding Neo4j...")
        
        # Clear existing data
        logger.info("  Clearing existing data...")
        self.neo4j_client.execute_query("MATCH (n) DETACH DELETE n")
        logger.info("  ✓ Cleared existing data")
        
        for profile in profiles:
            # Create Applicant node
            self.neo4j_client.execute_write("""
                CREATE (a:Applicant:Person {
                    id: $id,
                    name: $name,
                    emirates_id: $emirates_id,
                    age: $age,
                    nationality: $nationality
                })
            """, {
                "id": profile["_applicant_id"],
                "name": profile["name"],
                "emirates_id": profile["emirates_id"],
                "age": profile["age"],
                "nationality": profile["nationality"],
            })
            
            # Create Address node
            addr_hash = hashlib.md5(profile["emirate"].encode()).hexdigest()[:16]
            self.neo4j_client.execute_write("""
                MERGE (addr:Address {hash: $hash})
                SET addr.emirate = $emirate, addr.country = 'UAE'
            """, {
                "hash": addr_hash,
                "emirate": profile["emirate"],
            })
            
            # Create LIVES_AT relationship
            self.neo4j_client.execute_write("""
                MATCH (a:Applicant {id: $applicant_id})
                MATCH (addr:Address {hash: $addr_hash})
                CREATE (a)-[:LIVES_AT {source: 'emirates_id'}]->(addr)
            """, {
                "applicant_id": profile["_applicant_id"],
                "addr_hash": addr_hash,
            })
            
            # Create Employer node and relationship (if employed)
            if profile["employer"]:
                self.neo4j_client.execute_write("""
                    MERGE (e:Employer {name: $employer_name})
                """, {
                    "employer_name": profile["employer"],
                })
                
                self.neo4j_client.execute_write("""
                    MATCH (a:Applicant {id: $applicant_id})
                    MATCH (e:Employer {name: $employer_name})
                    CREATE (a)-[:WORKS_AT {
                        duration_months: $months,
                        source: 'resume'
                    }]->(e)
                """, {
                    "applicant_id": profile["_applicant_id"],
                    "employer_name": profile["employer"],
                    "months": profile["employment_months"],
                })
            
            # Create family member nodes (simplified - just count)
            for i in range(profile["num_dependents"]):
                self.neo4j_client.execute_write("""
                    MATCH (a:Applicant {id: $applicant_id})
                    CREATE (f:Person {
                        id: $family_id,
                        name: $family_name,
                        relationship: 'dependent'
                    })
                    CREATE (a)-[:FAMILY_OF {source: 'application_form'}]->(f)
                """, {
                    "applicant_id": profile["_applicant_id"],
                    "family_id": f"{profile['_applicant_id']}_dep{i+1}",
                    "family_name": f"Dependent {i+1}",
                })
            
            logger.debug(f"  ✓ Seeded graph for {profile['name']}")
        
        logger.info(f"✓ Neo4j seeded with {len(profiles)} profiles")
        
        # Print graph statistics
        node_count = self.neo4j_client.execute_query("MATCH (n) RETURN count(n) as count")[0]["count"]
        rel_count = self.neo4j_client.execute_query("MATCH ()-[r]->() RETURN count(r) as count")[0]["count"]
        logger.info(f"  Graph stats: {node_count} nodes, {rel_count} relationships")


async def main():
    """Main seeding workflow."""
    seeder = DatabaseSeeder()
    
    try:
        logger.info("=" * 60)
        logger.info("Database Seeding")
        logger.info("=" * 60)
        
        # Connect to all databases
        await seeder.connect_all()
        
        # Load data
        profiles = seeder.load_profiles()
        programs = seeder.load_programs()
        
        # Seed each database
        await seeder.seed_postgres(profiles)
        seeder.seed_mongo(profiles)
        seeder.seed_qdrant(profiles, programs)
        seeder.seed_neo4j(profiles)
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ All databases seeded successfully!")
        logger.info("=" * 60)
        logger.info("\nNext step: python -m scripts.verify_databases")
        
    finally:
        await seeder.disconnect_all()


if __name__ == "__main__":
    asyncio.run(main())
