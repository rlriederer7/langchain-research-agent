import pytest
from services.llm_service import LLMService
from unittest.mock import patch


class TestLLMService:
    @patch('services.llm_service.settings.LLM_PROVIDER', 'anthropic')
    @patch('services.llm_service.settings.MODEL_NAME', 'claude-sonnet-4-5-20250929')
    def test_creates_anthropic_llm(self):
        service = LLMService()
        llm = service.get_llm()

        from langchain_anthropic import ChatAnthropic
        assert isinstance(llm, ChatAnthropic)

    @patch('services.llm_service.settings.LLM_PROVIDER', 'ollama')
    @patch('services.llm_service.settings.MODEL_NAME', 'llama3.1:8b')
    def test_creates_ollama_llm(self):
        service = LLMService()
        llm = service.get_llm()

        from langchain_ollama import ChatOllama
        assert isinstance(llm, ChatOllama)

    @patch('services.llm_service.settings.LLM_PROVIDER', 'anthropic')
    def test_custom_temperature_anthropic(self):
        service = LLMService()
        llm = service.get_llm(temperature=0.5)

        assert llm.temperature == 0.5

    @patch('services.llm_service.settings.LLM_PROVIDER', 'ollama')
    def test_custom_temperature_ollama(self):
        service = LLMService()
        llm = service.get_llm(temperature=0.5)

        assert llm.temperature == 0.5
