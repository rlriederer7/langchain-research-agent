from typing import Optional, Dict, List, Any
from pydantic import BaseModel


class GraphAgentRequest(BaseModel):
    query: str
    max_iterations: Optional[int] = 10
    namespace: Optional[str] = None
    thread_id: Optional[str] = None


class GraphAgentResponse(BaseModel):
    response: List[Dict]
