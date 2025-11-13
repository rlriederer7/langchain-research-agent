import os
import tempfile
from pathlib import Path
from typing import List, BinaryIO, Optional, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, CSVLoader
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from services.pinecone_vector_service import pinecone_vector_service


class DocumentProcessorService:
    def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 200
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_document(
            self,
            file_data: bytes | BinaryIO,
            filename: str
    ) -> List[Document]:
        ext = Path(filename).suffix.lower()

        loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader,
            '.csv': CSVLoader
        }

        loader_class = loaders.get(ext)
        if not loader_class:
            raise ValueError(f"Unsupported file type: {ext}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            if isinstance(file_data, bytes):
                tmp_file.write(file_data)
            else:
                tmp_file.write(file_data.read())
            tmp_path = tmp_file.name

        try:
            loader = loader_class(tmp_path)
            documents = loader.load()
        finally:
            os.unlink(tmp_path)

        return documents

    def process_documents(
            self,
            documents: List[Document],
            metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        chunks = self.text_splitter.split_documents(documents)

        if metadata:
            for chunk in chunks:
                chunk.metadata.update(metadata)

        return chunks

    def load_and_process(
            self,
            file_data: bytes | BinaryIO,
            filename: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        documents = self.load_document(file_data, filename)
        return self.process_documents(documents, metadata)


class DocumentVectorPipeline:

    def __init__(
            self,
            processor: DocumentProcessorService,
            vector_service: PineconeVectorStore
    ):
        self.processor = processor
        self.vector_service = vector_service

    def process_and_upload(
            self,
            file_data: bytes | BinaryIO,
            filename: str,
            namespace: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> PineconeVectorStore:
        chunks = self.processor.load_and_process(file_data, filename, metadata)

        vectorstore = self.vector_service.upload_documents(chunks, namespace)

        return vectorstore


document_processor_service = DocumentProcessorService()
document_vector_pipeline = DocumentVectorPipeline(
    processor=document_processor_service,
    vector_service=pinecone_vector_service
)
