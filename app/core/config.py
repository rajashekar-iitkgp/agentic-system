from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Scalable Agentic API"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False

    GEMINI_API_KEY: Optional[str] = None
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_PROJECT: str = "scalable_agent_system"

    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    PINECONE_INDEX_NAME: str = "tool-registry"

    REDIS_URL: str = "redis://localhost:6379/0"
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/agent_db"

    
    PAYPAL_CLIENT_ID: Optional[str] = None
    PAYPAL_CLIENT_SECRET: Optional[str] = None
    PAYPAL_MODE: str = "sandbox"  # sandbox or live


    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()
