from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from core.config import settings


class LLMService:
    def __init__(self):
        self.llm = self.create_llm()

    def create_llm(self, temperature=None, max_tokens=None):
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        max_tok = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS

        if settings.LLM_PROVIDER == "anthropic":
            return ChatAnthropic(
                model=settings.ANTHROPIC_MODEL_NAME,
                temperature=temp,
                max_tokens=max_tok,
                max_retries=settings.LLM_MAX_RETRIES,
                timeout=settings.LLM_MAX_TIMEOUT,
            )
        elif settings.LLM_PROVIDER == "ollama":
            return ChatOllama(
                model=settings.OLLAMA_MODEL_NAME,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=temp,
                num_predict=max_tok,
            )
        else:
            raise ValueError(f"Unhandled LLM provider: {settings.LLM_PROVIDER}")

    def get_llm(self, temperature=None, max_tokens=None):
        return self.create_llm(temperature, max_tokens)


llm_service = LLMService()
