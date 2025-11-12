from langchain_anthropic import ChatAnthropic
from core.config import settings


class LLMService:
    def __init__(self):
        self.llm = ChatAnthropic(
            model=settings.MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            max_retries=settings.LLM_MAX_RETRIES,
            timeout=settings.LLM_MAX_TIMEOUT,
        )

    def get_llm(self, temperature=None, max_tokens=None):
        return ChatAnthropic(
            model=settings.MODEL_NAME,
            temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
            max_tokens=max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS,
            max_retries=settings.LLM_MAX_RETRIES,
            timeout=settings.LLM_MAX_TIMEOUT,
        )


llm_service = LLMService()
