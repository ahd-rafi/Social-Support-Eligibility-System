"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.db.postgres import PostgresClient
from src.db.mongo import MongoClient
from src.db.neo4j_client import Neo4jClient
from src.db.qdrant_client import QdrantClientWrapper

logger = logging.getLogger(__name__)

# Global database clients
postgres_client = PostgresClient()
mongo_client = MongoClient()
neo4j_client = Neo4jClient()
qdrant_client = QdrantClientWrapper()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    logger.info("Starting FastAPI application...")
    
    try:
        # Connect to databases
        logger.info("Connecting to databases...")
        await postgres_client.connect()
        mongo_client.connect()
        neo4j_client.connect()
        qdrant_client.connect()
        logger.info("✓ All databases connected")
        
    except Exception as e:
        logger.error(f"Failed to connect to databases: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI application...")
    
    try:
        await postgres_client.disconnect()
        mongo_client.disconnect()
        neo4j_client.disconnect()
        qdrant_client.disconnect()
        logger.info("✓ All databases disconnected")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


def create_app() -> FastAPI:
    """Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="Social Support Eligibility System API",
        description="AI-powered application processing for government social support programs",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify allowed origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.app_env == "development" else "An error occurred",
            },
        )
    
    # Include routers
    from src.api.routes import router
    app.include_router(router, prefix="/api")
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": "Social Support Eligibility System API",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs",
        }
    
    logger.info("FastAPI application created")
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.app_env == "development",
    )
