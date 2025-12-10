from typing import List, Optional, Dict, Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from agents.a_graph_base_agent import GraphBaseAgent
from services.llm_service import llm_service


class GraphChatAgent(GraphBaseAgent):
    DEFAULT_SYSTEM_PROMPT = """You are a helpful chatbot :)
        You have access to web search tools. Use them to find accurate, up-to-date information if you want to.
        When you find relevant information, cite your sources.
        Have fun :)"""

    def __init__(
            self,
            tools: List[BaseTool],
            vector_retriever,
            pinecone_index=None,
            llm: Optional[BaseLanguageModel] = None,
            max_iterations: int = 6,
            verbose: bool = True,
            session_id: Optional[str] = None,
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
            session_id=session_id,
            storage_adapter=storage_adapter,
        )
        print("finished agent init")

    async def research(self, query: str) -> Dict[str, Any]:
        return await self.run(query)


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
        session_id=session_id,
        storage_adapter=storage_adapter,
        **kwargs
    )
