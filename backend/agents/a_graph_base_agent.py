import json
from typing import Optional, List, Dict, Any, Literal

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from services.llm_service import llm_service
from agents.state import AgentState


class GraphBaseAgent:

    def __init__(
            self,
            tools: List[BaseTool],
            system_prompt: str,
            pinecone_index=None,
            llm: Optional[BaseLanguageModel] = None,
            max_iterations: int = 2,
            memory_config: Optional[Dict] = None,
            session_id: Optional[str] = None,
            storage_adapter=None,
    ):
        self.llm = llm or llm_service.get_llm()
        self.tools = tools
        self.max_iterations = max_iterations
        self.pinecone_index = pinecone_index
        self.session_id = session_id
        self.storage_adapter = storage_adapter

        self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.app = self._build_graph()

        print("finished agent init")

    def _call_model(self, state: AgentState):
        messages = state["messages"]
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> Literal["tools", "end"]:
        messages = state["messages"]
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self.tools))

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    async def run(self, query: str):
        from langchain_core.messages import HumanMessage
        inputs = {"messages": [HumanMessage(content=query)]}
        return await self.app.ainvoke(inputs)