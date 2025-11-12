from typing import Optional, List, Dict, Any
from langchain_anthropic import ChatAnthropic
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from services.llm_service import llm_service


class ResearchAgent:
    print(1.1)
    DEFAULT_SYSTEM_PROMPT = """You are a helpful research assistant.
        You have access to web search tools. Use them to find accurate, up-to-date information.
        When you find relevant information, cite your sources.
        Be thorough but concise in your research.
        Simple questions should beget simple results."""
    print(1.2)
    def __init__(
            self,
            tools: List[BaseTool],
            llm: Optional[BaseLanguageModel] = None,
            max_iterations: int = 10,
            verbose: bool = True,
    ):
        print(1.3)
        self.llm = llm_service.get_llm()
        print(1.4)
        self.tools = tools
        print(1.5)
        self.max_iterations = max_iterations
        print(1.6)
        self.verbose = verbose
        print(1.7)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        print(1.8)
        print(self.llm)
        print(self.tools)
        print(self.prompt)
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        print(1.9)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            return_intermediate_steps=True,
        )
        print("finished agent init")

    async def research(
            self,
            query: str,
            system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        print(1.11)
        print(query)
        print(system_prompt or self.DEFAULT_SYSTEM_PROMPT)
        result = await self.agent_executor.ainvoke({
            "input": query,
            "system_prompt": system_prompt or self.DEFAULT_SYSTEM_PROMPT
        })
        print(1.12)
        return {
            "output": result["output"],
            "intermediate_steps": [
                {
                    "tool": step[0].tool,
                    "tool_input": step[0].tool_input,
                    "tool_output": step[1]
                }
                for step in result.get("intermediate_steps", [])
            ]
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
