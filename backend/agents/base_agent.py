from typing import Optional, List, Dict, Any

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_classic.memory import ConversationBufferMemory, ConversationEntityMemory, VectorStoreRetrieverMemory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.tools import BaseTool

from services.llm_service import llm_service


def get_pinecone_namespaces(index) -> List[str]:
    stats = index.describe_index_stats()
    namespaces = list(stats.get('namespaces', {}).keys())
    print(namespaces)
    namespaces = [ns for ns in namespaces if ns]
    print(namespaces)
    return namespaces


class BaseAgent:

    def __init__(
            self,
            tools: List[BaseTool],
            system_prompt: str,
            pinecone_index=None,
            llm: Optional[BaseLanguageModel] = None,
            max_iterations: int = 2,
            verbose: bool = True,
            memory_config: Optional[Dict] = None,
    ):
        self.llm = llm or llm_service.get_llm()
        self.tools = tools
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.system_prompt = system_prompt
        self.pinecone_index = pinecone_index

        self.memories = self.setup_memory(memory_config) if memory_config else {}

        self.prompt = self.build_prompt(system_prompt)

        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
        )
        print("finished agent init")

    def setup_memory(self, config: Dict) -> Dict:
        memories = {}

        if config.get('vector_retriever'):
            memories['vector'] = VectorStoreRetrieverMemory(
                retriever=config['vector_retriever'],
                memory_key="long_term_context",
            )

        return memories

    def build_prompt(self, system_prompt: str) -> ChatPromptTemplate:
        messages = [("system", system_prompt)]

        if 'vector' in self.memories:
            messages.append(("system", "Relevant past context:\n{long_term_context}"))

        messages.extend([
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        return ChatPromptTemplate.from_messages(messages)

    def load_memories(self, query: str) -> Dict[str, Any]:
        memory_vars = {}

        if 'vector' in self.memories:
            memory_vars.update(
                self.memories['vector'].load_memory_variables({"prompt": query})
            )

        return memory_vars

    def save_to_memories(self, input_text: str, output_text: str):
        context = {"input": input_text, "output": output_text}

        if 'vector' in self.memories:
            self.memories['vector'].save_context(
                {"input": input_text},
                {"output": output_text}
            )

    async def run(self, query: str) -> Dict[str, Any]:
        print(2.1)
        memory_vars = self.load_memories(query)
        print(2.2)
        result = await self.agent_executor.ainvoke({
            "input": query,
            **memory_vars
        })

        self.save_to_memories(query, result["output"])

        print(f"Result keys: {result.keys()}")
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")

        return {
            "output": result.get("output", "")
        }
