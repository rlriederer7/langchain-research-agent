from fastapi import APIRouter, HTTPException
from agents.chat_agent import create_chat_agent
from agents.research_agent import create_research_agent
from chains.query_decomposition_chain import QueryDecompositionChain
from core.config import settings
from core.context_vars import request_namespace
from models.agent_models import AgentResponse, AgentRequest
from services.llm_service import llm_service
from services.pinecone_vector_service import pinecone_vector_service
from storage_adapters.file_storage_adapter import FileStorageAdapter
from tools.retriever import retrieve_context
from tools.web_search import get_search_web_ddg

router = APIRouter()


@router.post("/chat_agentically", response_model=AgentResponse)
async def chat_agent(request: AgentRequest):
    try:
        tools = [get_search_web_ddg(), retrieve_context]
        storage = FileStorageAdapter()

        vector_retriever = pinecone_vector_service.get_vectorstore(
            namespace=request.namespace
        ).as_retriever()

        agent = create_chat_agent(
            max_iterations=request.max_iterations,
            tools=tools,
            pinecone_index=pinecone_vector_service.pc.Index(settings.PINECONE_INDEX_NAME),
            vector_retriever=vector_retriever,
            session_id=request.session_id,
            storage_adapter=storage,
        )

        result = await agent.research(
            query=request.query
        )

        return AgentResponse(
            response=result["output"][0]['text']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in agent execution: {str(e)}")


@router.post("/research", response_model=AgentResponse)
async def research_agent(request: AgentRequest):
    try:
        namespace = request.namespace
        request_namespace.set(namespace)

        tools = [get_search_web_ddg(), retrieve_context]

        agent = create_research_agent(
            max_iterations=request.max_iterations,
            tools=tools,
            pinecone_index=pinecone_vector_service.pc.Index(settings.PINECONE_INDEX_NAME),
        )

        result = await agent.research(
            query=request.query
        )

        return AgentResponse(
            response=result["output"][0]['text']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in agent execution: {str(e)}")


@router.post("/research_harder", response_model=AgentResponse)
async def research_agent_subquery(request: AgentRequest):
    try:
        namespace = request.namespace
        request_namespace.set(namespace)

        tools = [get_search_web_ddg(), retrieve_context]
        agent = create_research_agent(
            max_iterations=request.max_iterations,
            tools=tools,
            pinecone_index=pinecone_vector_service.pc.Index(settings.PINECONE_INDEX_NAME),
        )

        decomp_chain = QueryDecompositionChain(
            llm=llm_service.get_llm(),
            research_agent=agent
        )

        result = await decomp_chain.arun(request.query)

        return AgentResponse(
            response=result["final_answer"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in agent execution: {str(e)}")
