"""Application configuration using Pydantic Settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # Application
    app_env: Literal["development", "production"] = "development"
    upload_dir: Path = Path("./uploads")
    model_dir: Path = Path("./models")
    log_level: str = "INFO"

    # PostgreSQL
    postgres_user: str = "social_support"
    postgres_password: str
    postgres_db: str = "social_support"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    langfuse_db: str = "langfuse"

    @property
    def postgres_url(self) -> str:
        """PostgreSQL connection URL for asyncpg."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_url_sync(self) -> str:
        """PostgreSQL connection URL for psycopg (sync)."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # MongoDB
    mongo_initdb_root_username: str = "root"
    mongo_initdb_root_password: str
    mongo_host: str = "localhost"
    mongo_port: int = 27017
    mongo_db: str = "social_support"

    @property
    def mongo_url(self) -> str:
        """MongoDB connection URL."""
        from urllib.parse import quote_plus
        # URL-encode username and password to handle special characters
        username = quote_plus(self.mongo_initdb_root_username)
        password = quote_plus(self.mongo_initdb_root_password)
        return (
            f"mongodb://{username}:{password}"
            f"@{self.mongo_host}:{self.mongo_port}/"
        )

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334

    @property
    def qdrant_url(self) -> str:
        """Qdrant HTTP endpoint."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str
    neo4j_database: str = "neo4j"

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "phi4-mini"

    # Langfuse
    langfuse_secret_key: str = Field(default="")
    langfuse_public_key: str = Field(default="")
    langfuse_host: str = "http://localhost:3000"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Streamlit
    streamlit_server_port: int = 8501


# Global settings instance
settings = Settings()


# Create directories if they don't exist
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.model_dir.mkdir(parents=True, exist_ok=True)
