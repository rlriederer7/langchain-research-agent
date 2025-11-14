from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    ANTHROPIC_API_KEY: str
    MODEL_NAME: str = "claude-haiku-4-5-20251001"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 1024
    LLM_MAX_RETRIES: int = 3
    LLM_MAX_TIMEOUT: float = 60.0

    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "pinecone-index-2"
    PINECONE_ENVIRONMENT: str = "us-east-1"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()