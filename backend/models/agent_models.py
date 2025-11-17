from typing import Optional

from pydantic import BaseModel


class AgentRequest(BaseModel):
    query: str
    max_iterations: Optional[int] = 10
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    namespace: Optional[str] = None
    session_id: Optional[str] = None


class AgentResponse(BaseModel):
    response: str
