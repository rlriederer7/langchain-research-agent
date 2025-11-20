import pytest
import tempfile
from unittest.mock import MagicMock, AsyncMock, Mock
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Any, Sequence
from langchain_core.runnables import Runnable


class FakeLLM(BaseChatModel):
    responses: List[str] = ["this is a first test response."]
    call_count: int = 0

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: List[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
    ) -> ChatResult:
        """return a fake response, cycling through test response list"""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1

        message = AIMessage(content=response)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
            self,
            messages: List[BaseMessage],
            stop: List[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
    ) -> ChatResult:
        """async _generate"""
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(
            self,
            tools: Sequence[Any],
            **kwargs: Any,
    ) -> Runnable[Any, BaseMessage]:
        return self


class TestBaseAgent:

    @pytest.fixture
    def fake_llm(self):
        return FakeLLM()

    @pytest.fixture
    def temp_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from storage_adapters.file_storage_adapter import FileStorageAdapter
            yield FileStorageAdapter(storage_dir=tmpdir)

    @pytest.fixture
    def agent_with_fake_llm(self, fake_llm, temp_storage):
        from agents.base_agent import BaseAgent

        return BaseAgent(
            tools=[],
            system_prompt="You are a helpful assistant",
            llm=fake_llm,
            session_id="test_session",
            storage_adapter=temp_storage,
            memory_config={'short_term': True}
        )

    @pytest.mark.asyncio
    async def test_agent_basic_query(self, agent_with_fake_llm):
        """basic agent query"""
        result = await agent_with_fake_llm.run("Hello, how are you?")

        assert "output" in result
        assert isinstance(result["output"], str)
        assert len(result["output"]) > 0
        assert result["output"] == "this is a first test response."
        print(f"Agent output: {result['output']}")

    @pytest.mark.asyncio
    async def test_agent_saves_to_memory(self, agent_with_fake_llm, temp_storage):
        """agent saves conversation to short term"""
        await agent_with_fake_llm.run("Test query")

        saved_data = temp_storage.load("test_session")
        assert saved_data is not None
        assert "Test query" in saved_data

    @pytest.mark.asyncio
    async def test_agent_loads_previous_conversation(self, temp_storage, fake_llm):
        """agent loads short term"""
        import json
        from langchain_core.messages import messages_to_dict
        from agents.base_agent import BaseAgent

        existing_messages = [
            HumanMessage(content="Previous question"),
            AIMessage(content="Previous answer")
        ]
        temp_storage.save("test_session", json.dumps(messages_to_dict(existing_messages)))

        agent = BaseAgent(
            tools=[],
            system_prompt="You are a helpful assistant",
            llm=fake_llm,
            session_id="test_session",
            storage_adapter=temp_storage,
            memory_config={'short_term': True}
        )

        memory_vars = agent.load_memory("New question")
        assert "chat_history" in memory_vars
        assert len(memory_vars["chat_history"]) == 2
        assert memory_vars["chat_history"][0].content == "Previous question"
        assert memory_vars["chat_history"][1].content == "Previous answer"

    @pytest.mark.asyncio
    async def test_agent_multiple_queries(self, agent_with_fake_llm):
        """Test multiple sequential queries"""
        result1 = await agent_with_fake_llm.run("First query")
        assert result1["output"] == "this is a first test response."

        result2 = await agent_with_fake_llm.run("Second query")
        assert result2["output"] == "this is a first test response."

    @pytest.mark.asyncio
    async def test_fake_llm_cycling_responses(self):
        """FakeLLM multiple responses/cycling"""
        fake_llm = FakeLLM(responses=["Response 1", "Response 2", "Response 3"])

        messages = [HumanMessage(content="test")]

        result1 = await fake_llm._agenerate(messages)
        assert result1.generations[0].text == "Response 1"

        result2 = await fake_llm._agenerate(messages)
        assert result2.generations[0].text == "Response 2"

        result3 = await fake_llm._agenerate(messages)
        assert result3.generations[0].text == "Response 3"

        # Should cycle back to first response
        result4 = await fake_llm._agenerate(messages)
        assert result4.generations[0].text == "Response 1"

    @pytest.mark.asyncio
    async def test_agent_conversation_continuity(self, temp_storage, fake_llm):
        """conversation history across multiple queries"""
        from agents.base_agent import BaseAgent

        agent = BaseAgent(
            tools=[],
            system_prompt="You are a helpful assistant",
            llm=fake_llm,
            session_id="continuity_test",
            storage_adapter=temp_storage,
            memory_config={'short_term': True}
        )

        await agent.run("What is Python?")

        memory_vars = agent.load_memory("Tell me more")
        assert len(memory_vars["chat_history"]) == 2

    @pytest.mark.asyncio
    async def test_agent_with_custom_responses(self, temp_storage):
        """agent with response sequence"""
        custom_llm = FakeLLM(responses=[
            "First response",
            "Second response",
            "Third response"
        ])

        from agents.base_agent import BaseAgent
        agent = BaseAgent(
            tools=[],
            system_prompt="You are a helpful assistant",
            llm=custom_llm,
            session_id="custom_test",
            storage_adapter=temp_storage,
            memory_config={'short_term': True}
        )

        result1 = await agent.run("Query 1")
        assert result1["output"] == "First response"

        result2 = await agent.run("Query 2")
        assert result2["output"] == "Second response"

        result3 = await agent.run("Query 3")
        assert result3["output"] == "Third response"

        result3 = await agent.run("Query 4")
        assert result3["output"] == "First response"
