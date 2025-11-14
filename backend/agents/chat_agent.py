from typing import Optional, List, Dict, Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from agents.base_agent import BaseAgent
from services.llm_service import llm_service


class ChatAgent(BaseAgent):
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
    ):
        super().__init__(
            tools=tools,
            system_prompt=self.DEFAULT_SYSTEM_PROMPT,
            pinecone_index=pinecone_index,
            llm=llm,
            max_iterations=max_iterations,
            verbose=verbose,
            memory_config={
                'short_term': True,
                'entity': True,
                'vector_retriever': vector_retriever
            },
        )
        print("finished agent init")

    async def research(self, query: str) -> Dict[str, Any]:
        return await self.run(query)


def create_chat_agent(
        tools: List[BaseTool],
        vector_retriever,
        pinecone_index,
        llm: Optional[BaseLanguageModel] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
) -> ChatAgent:
    chat_llm = llm or llm_service.get_llm(
        temperature=temperature,
        max_tokens=max_tokens
    )
    return ChatAgent(
        llm=chat_llm,
        tools=tools,
        vector_retriever=vector_retriever,
        pinecone_index=pinecone_index,
        **kwargs
    )
