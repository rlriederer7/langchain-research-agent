from typing import Optional

from pydantic import BaseModel


class AgentRequest(BaseModel):
    query: str
    system_prompt: Optional[str] = None
    max_iterations: Optional[int] = 10
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class AgentResponse(BaseModel):
    response: str
