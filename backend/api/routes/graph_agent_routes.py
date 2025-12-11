from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

from agents.a_graph_chat_agent import create_graph_chat_agent
from core.config import settings
from core.context_vars import request_namespace
from models.graph_agent_models import GraphAgentRequest, GraphAgentResponse
from services.pinecone_vector_service import pinecone_vector_service
from storage_adapters.file_storage_adapter import FileStorageAdapter
from tools.retriever import retrieve_context
from tools.web_search import get_search_web_ddg

router = APIRouter()


@router.post("/chat_graph_agentically", response_model=GraphAgentResponse)
async def chat_graph_agent(request: GraphAgentRequest):
    try:
        namespace = request.namespace
        request_namespace.set(namespace)
        print(request.namespace)

        tools = [get_search_web_ddg(), retrieve_context]
        storage = FileStorageAdapter()

        vector_retriever = pinecone_vector_service.get_vectorstore(
            namespace=request.namespace
        ).as_retriever()

        chat_agent = create_graph_chat_agent(
            max_iterations=request.max_iterations,
            tools=tools,
            pinecone_index=pinecone_vector_service.pc.Index(settings.PINECONE_INDEX_NAME),
            vector_retriever=vector_retriever,
            storage_adapter=storage,
        )

        result = await chat_agent.chat(
            query=request.query,
            thread_id=request.thread_id
        )

        messages = result["messages"]
        response = []

        for message in messages:
            if isinstance(message, AIMessage):
                if message.content:
                    response.append({"type": "ai", "content": message.content})

                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tool_call in message.tool_calls:
                        response.append({
                            "type": "tool_call",
                            "tool": tool_call["name"],
                            "args": tool_call["args"]
                        })

            elif isinstance(message, ToolMessage):
                response.append({
                    "type": "tool_result",
                    "tool": message.name,
                    "result": message.content
                })

            elif isinstance(message, HumanMessage):
                response.append({"type": "human", "content": message.content})

        print(type(result))
        print(result)
        print(type(messages))
        print(messages)
        print(type(response))
        print(response)
        return GraphAgentResponse(response=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in agent execution: {str(e)}")
