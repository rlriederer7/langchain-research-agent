from pydantic_settings import BaseSettings
from typing import Optional, Literal


class Settings(BaseSettings):
    LLM_PROVIDER: Literal["anthropic"] = "anthropic"
    ANTHROPIC_MODEL_NAME: str = "claude-haiku-4-5-20251001"
    OLLAMA_MODEL_NAME: str = "llama3.1:8b"
    OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 1024
    LLM_MAX_RETRIES: int = 3
    LLM_MAX_TIMEOUT: float = 60.0

    ANTHROPIC_API_KEY: Optional[str] = None

    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: str = "pinecone-index-2"
    PINECONE_ENVIRONMENT: str = "us-east-1"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()