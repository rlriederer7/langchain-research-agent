from typing import List, Optional, Any, Dict

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from core.config import settings


class PineconeVectorService:
    def __init__(
            self,
            index_name: str = settings.PINECONE_INDEX_NAME,
    ):
        self.region = settings.PINECONE_ENVIRONMENT
        self.index_name = index_name
        if not settings.PINECONE_API_KEY:
            raise ValueError("Pinecone API key not found")
        self.pc = PineconeClient(api_key=settings.PINECONE_API_KEY)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.ensure_index_exists()

    def ensure_index_exists(self):
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            sample_text = "dimension_text"
            embedding_vector = self.embeddings.embed_query(sample_text)
            dimension = len(embedding_vector)

            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.region
                )
            )

    def upload_documents(
            self,
            documents: List[Document],
            namespace: Optional[str] = None
    ) -> PineconeVectorStore:
        vectorstore = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
            index_name=self.index_name,
            namespace=namespace,
        )
        return vectorstore

    def get_vectorstore(
            self,
            namespace: Optional[str] = None
    ) -> PineconeVectorStore:
        return PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace=namespace
        )

    def similarity_search(
            self,
            query: str,
            k: int = 10,
            namespace: Optional[str] = None,
            filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        vectorstore = self.get_vectorstore(namespace)
        return vectorstore.similarity_search(
            query,
            k=k,
            filter=filter
        )

    def delete_namespace(
            self,
            namespace: str
    ):
        index = self.pc.Index(self.index_name)
        index.delete(delete_all=True, namespace=namespace)


pinecone_vector_service = PineconeVectorService()
