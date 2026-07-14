"""Clear all data from PostgreSQL database."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db.postgres import PostgresClient

async def clear_database():
    """Clear all application data from PostgreSQL."""
    client = PostgresClient()
    
    try:
        await client.connect()
        print("Clearing database...")
        
        # TRUNCATE CASCADE will remove all related records
        await client.execute("TRUNCATE TABLE applicants CASCADE")
        
        print("✓ Database cleared successfully!")
        print("  - All applicants removed")
        print("  - All applications removed")
        print("  - All features removed")
        print("  - All decisions removed")
        
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(clear_database())
