from fastapi import APIRouter, HTTPException
import json
from agents.research_agent import create_research_agent
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
            response=result["output"][0]['text'],
            intermediate_steps=result["intermediate_steps"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in agent execution: {str(e)}")
