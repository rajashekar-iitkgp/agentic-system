from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized configuration for the Agentic System.
    Loads from environment variables or a .env file.
    """
    # API Settings
    PROJECT_NAME: str = "Scalable Agentic API"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False

    # LLM Settings (Google Gemini)
    GEMINI_API_KEY: Optional[str] = None
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_PROJECT: str = "scalable_agent_system"

    # Pinecone Vector DB
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    PINECONE_INDEX_NAME: str = "tool-registry"

    # PostgreSQL (For audit, logs, persistence)
    DATABASE_URL: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()
