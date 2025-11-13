from typing import Optional
from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from models.document_models import DocumentUploadResponse, DocumentSearchResponse, DocumentSearchRequest, DocumentChunk, \
    NamespaceDeleteResponse
from services.document_service import document_vector_pipeline
from services.pinecone_vector_service import pinecone_vector_service
import json

router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
        file: UploadFile = File(...),
        namespace: Optional[str] = Form(None),
        metadata: Optional[str] = Form(None)
):
    try:
        file_bytes = await file.read()
        meta_dict = None
        if metadata:
            meta_dict = json.loads(metadata)

        vectorstore = document_vector_pipeline.process_and_upload(
            file_data=file_bytes,
            filename=file.filename,
            namespace=namespace,
            metadata=meta_dict
        )

        return DocumentUploadResponse(
            message="Doc uploaded",
            namespace=namespace,
            document_count=len(vectorstore._texts) if hasattr(vectorstore, "_texts") else 0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")


@router.post("/search", response_model=DocumentSearchResponse)
async def search_documents(request: DocumentSearchRequest):
    try:
        results = pinecone_vector_service.similarity_search(
            query=request.query,
            k=request.k,
            namespace=request.namespace,
            filter=request.filter,
        )
        return DocumentSearchResponse(
            results=[
                DocumentChunk(
                    page_content=doc.page_content,
                    metadata=doc.metadata
                )
                for doc in results
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {e}")


@router.delete("/namespace/{namespace}", response_model=NamespaceDeleteResponse)
async def delete_namespace(namespace: str):
    try:
        pinecone_vector_service.delete_namespace(namespace)
        return NamespaceDeleteResponse(
            message="Namespace deleted :)",
            namespace=namespace
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting namespace: {e}")
