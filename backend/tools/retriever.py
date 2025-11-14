from langchain_core.tools import tool

from core.context_vars import request_namespace
from services.pinecone_vector_service import pinecone_vector_service


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant context from the vector database based on the query."""
    namespace = request_namespace.get()
    print(namespace)
    retrieved_docs = pinecone_vector_service.similarity_search(query, k=3, namespace=namespace)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
