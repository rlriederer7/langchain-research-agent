from typing import Optional, List, Dict, Any
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from services.llm_service import llm_service


class ResearchAgent:
    DEFAULT_SYSTEM_PROMPT = """You are a helpful research assistant.
        You have access to web search tools. Use them to find accurate, up-to-date information.
        When you find relevant information, cite your sources.
        Be thorough but concise in your research.
        You also have access to a tool that retrieves chunks from uploaded technical documents.
        Simple questions should beget simple results."""

    def __init__(
            self,
            tools: List[BaseTool],
            llm: Optional[BaseLanguageModel] = None,
            max_iterations: int = 2,
            verbose: bool = True,
    ):
        self.llm = llm_service.get_llm()
        self.tools = tools
        self.max_iterations = max_iterations
        self.verbose = verbose

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        print(self.llm)
        print(self.tools)
        print(self.prompt)

        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
        )
        print("finished agent init")

    async def research(
            self,
            query: str,
            system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        result = await self.agent_executor.ainvoke({
            "input": query,
            "system_prompt": system_prompt or self.DEFAULT_SYSTEM_PROMPT
        })

        print(f"Result keys: {result.keys()}")
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")

        return {
            "output": result.get("output", "")
        }


def create_research_agent(
        tools: List[BaseTool],
        llm: Optional[BaseLanguageModel] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
) -> ResearchAgent:
    research_llm = llm or llm_service.get_llm(temperature=temperature, max_tokens=max_tokens)
    return ResearchAgent(llm=research_llm, tools=tools, **kwargs)
