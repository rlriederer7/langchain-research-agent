from typing import List, Optional, Dict, Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from agents.a_graph_base_agent import GraphBaseAgent
from core.database import get_checkpointer
from services.llm_service import llm_service


class GraphChatAgent(GraphBaseAgent):
    # TODO: Implement system prompt for LangGraph agents. Currently does not use at any point.
    DEFAULT_SYSTEM_PROMPT = """You are a helpful chatbot :)
        You have access to web search tools. Use them to find accurate, up-to-date information if you want to.
        When you find relevant information, cite your sources.
        Have fun :)
        
        Remember, the user only ever sees your *last* message.
        If you respond to the user, use tools, and then finish responding, the user only sees 
        the second half of your response to the user."""

    def __init__(
            self,
            tools: List[BaseTool],
            vector_retriever,
            pinecone_index=None,
            llm: Optional[BaseLanguageModel] = None,
            max_iterations: int = 6,
            verbose: bool = True,
            storage_adapter=None,
    ):
        super().__init__(
            tools=tools,
            system_prompt=self.DEFAULT_SYSTEM_PROMPT,
            pinecone_index=pinecone_index,
            llm=llm,
            max_iterations=max_iterations,
            memory_config={
                'short_term': True,
                'vector_retriever': vector_retriever
            },
            storage_adapter=storage_adapter,
            checkpointer=get_checkpointer()
        )
        print("finished agent init")

    async def chat(self, query: str, thread_id: str) -> Dict[str, Any]:
        print(10)
        return await self.run(query, thread_id)


def create_graph_chat_agent(
        tools: List[BaseTool],
        vector_retriever,
        pinecone_index,
        llm: Optional[BaseLanguageModel] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        session_id: Optional[str] = None,
        storage_adapter=None,
        **kwargs
) -> GraphChatAgent:

    chat_llm = llm or llm_service.get_llm(
        temperature=temperature,
        max_tokens=max_tokens
    )

    return GraphChatAgent(
        llm=chat_llm,
        tools=tools,
        vector_retriever=vector_retriever,
        pinecone_index=pinecone_index,
        storage_adapter=storage_adapter,
        **kwargs
    )