from langchain_core.tools import tool

from services.pinecone_vector_service import pinecone_vector_service


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant context from the vector database based on the query."""
    retrieved_docs = pinecone_vector_service.similarity_search(query, k=3)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
