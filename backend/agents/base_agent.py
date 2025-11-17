import json
from typing import Optional, List, Dict, Any

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_classic.memory import ConversationBufferMemory, VectorStoreRetrieverMemory, CombinedMemory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import messages_from_dict, messages_to_dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from services.llm_service import llm_service


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
            session_id: Optional[str] = None,
            storage_adapter=None,
    ):
        self.llm = llm or llm_service.get_llm()
        self.tools = tools
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.system_prompt = system_prompt
        self.pinecone_index = pinecone_index
        self.session_id = session_id
        self.storage_adapter = storage_adapter

        self.memory = self.setup_memory(memory_config or {})

        self.prompt = self.build_prompt(system_prompt)

        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
        )
        print("finished agent init")

    def setup_memory(self, config: Dict) -> CombinedMemory:
        memories = []

        if config.get('short_term'):
            conversation_memory = ConversationBufferMemory(
                    return_messages=True,
                    memory_key="chat_history",
                    input_key="input",
                    output_key="output",
            )

            if self.session_id and self.storage_adapter:
                self.load_chat_history(conversation_memory)

            memories.append(conversation_memory)

        if config.get('vector_retriever'):
            memories.append(
                VectorStoreRetrieverMemory(
                    retriever=config['vector_retriever'],
                    memory_key="long_term_context",
                    input_key="input",
                )
            )

        return CombinedMemory(memories=memories)

    def load_chat_history(self, conversation_memory: ConversationBufferMemory):
        try:
            serialized_history = self.storage_adapter.load(self.session_id)
            if serialized_history:
                message_dicts = json.loads(serialized_history)
                messages = messages_from_dict(message_dicts)

                conversation_memory.chat_memory.messages = messages
                print(f"Loaded {len(messages)} messages from storage for session {self.session_id}")
        except Exception as e:
            print(f"Error loading chat history: {e}")

    def save_chat_history(self):
        if not self.session_id or not self.storage_adapter or not self.memory:
            return

        try:
            for mem in self.memory.memories:
                if isinstance(mem, ConversationBufferMemory):
                    messages = mem.chat_memory.messages
                    message_dicts = messages_to_dict(messages)
                    serialized = json.dumps(message_dicts)

                    self.storage_adapter.save(self.session_id, serialized)
                    print(f"saved {len(messages)} messages to storage for session {self.session_id}")
                    break
        except Exception as e:
            print(f"Error saving chat history: {e}")

    def build_prompt(self, system_prompt: str) -> ChatPromptTemplate:
        messages = [("system", system_prompt)]

        if self.memory:
            for mem in self.memory.memories:
                if getattr(mem, "memory_key", None) == "long_term_context":
                    messages.append(("system", "Relevant past context:\n{long_term_context}"))

        messages.append(("placeholder", "{chat_history}"))

        messages.extend([
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        print(messages)

        return ChatPromptTemplate.from_messages(messages)

    def load_memory(self, query: str) -> Dict[str, Any]:
        if not self.memory:
            return {}
        return self.memory.load_memory_variables({"input": query})

    def save_to_memory(self, input_text: str, output_text: str):
        if not self.memory:
            print("not self.memory")
            return
        if isinstance(output_text, list):
            output_str_text = " ".join([x.get("text","") for x in output_text])
        else:
            output_str_text = str(output_text)
        self.memory.save_context(
            {"input": input_text},
            {"output": output_str_text},
        )

        self.save_chat_history()

    async def run(self, query: str) -> Dict[str, Any]:
        vars = self.memory.load_memory_variables({"input": query})
        memory_vars = self.load_memory(query)
        result = await self.agent_executor.ainvoke({
            "input": query,
            **memory_vars
        })

        self.save_to_memory(query, result.get("output",""))

        print(f"Result keys: {result.keys()}")
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")

        return {
            "output": result.get("output", "")
        }
