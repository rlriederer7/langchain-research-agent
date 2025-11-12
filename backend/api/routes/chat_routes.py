import os
from fastapi import APIRouter, HTTPException
from models.chat_models import Message, ChatRequest, ChatResponse
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from services.chat_service import llm_service

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        lc_messages = []

        if request.system_prompt:
            lc_messages.append(SystemMessage(content=request.system_prompt))

        for msg in request.messages:
            if msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                lc_messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                lc_messages.append(SystemMessage(content=msg.content))

        llm_with_params = llm_service.get_llm(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        print(llm_service.llm.max_retries)
        response = await llm_with_params.ainvoke(lc_messages)

        return ChatResponse(
            response=response.content,
            model=llm_with_params.model
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")
