import asyncio

from fastapi import APIRouter, HTTPException
import json
from agents.research_agent import create_research_agent
from chains.query_decomposition_chain import QueryDecompositionChain
from models.agent_models import AgentResponse, AgentRequest
from tools.web_search import get_search_web_ddg
from tools.retriever import retrieve_context
from services.llm_service import llm_service

router = APIRouter()


@router.post("/research", response_model=AgentResponse)
async def research_agent(request: AgentRequest):
    try:
        tools = [get_search_web_ddg(), retrieve_context]
        agent = create_research_agent(max_iterations=request.max_iterations, tools=tools)

        result = await agent.research(
            query=request.query,
            system_prompt=request.system_prompt
        )

        return AgentResponse(
            response=result["output"][0]['text']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in agent execution: {str(e)}")


@router.post("/research_harder", response_model=AgentResponse)
async def research_agent_subquery(request: AgentRequest):
    try:
        tools = [get_search_web_ddg(), retrieve_context]
        agent = create_research_agent(max_iterations=request.max_iterations, tools=tools)

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
