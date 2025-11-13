from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    namespace: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentUploadResponse(BaseModel):
    message: str
    namespace: Optional[str] = None
    document_count: int


class DocumentSearchRequest(BaseModel):
    query: str
    k: Optional[int] = 10
    namespace: Optional[str] = None
    filter: Optional[Dict[str, Any]] = None


class DocumentChunk(BaseModel):
    page_content: str
    metadata: Optional[Dict[str, Any]] = None


class DocumentSearchResponse(BaseModel):
    results: List[DocumentChunk]


class NamespaceDeleteResponse(BaseModel):
    message: str
    namespace: str