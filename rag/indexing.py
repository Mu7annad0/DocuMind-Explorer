"""Indexing"""
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFium2Loader
from langchain_cohere import CohereEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever
from rag.config import Config


class Indexing:
    """The Indexing phase"""
    def __init__(self):
        self.embeddings = CohereEmbeddings(model="embed-english-v3.0")
        self.semantic_splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="interquartile"
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 2048,
            chunk_overlap = 100,
            add_start_index=True
        )

    def load_documents(self, doc_files: List[Path]) -> List[str]:
        """Load documents from the given list of file paths."""
        documents = []
        for doc_file in doc_files:
            loaded_docs = PyPDFium2Loader(doc_file).load()
            documents.extend([doc.page_content for doc in loaded_docs])
        return documents

    def index(self, doc_files: List[Path]) -> VectorStore:
        """
        Index phase containing of:
            1. load
            2. split
            3. embed
            4. store
        """
        documents = self.load_documents(doc_files)
        documents = self.recursive_splitter.split_documents(
            self.semantic_splitter.create_documents(["\n".join(documents)])
        )
        # Semantic Search using Qdrant Vector databae.
        vector_db = Qdrant.from_documents(
            documents = documents,
            embedding=self.embeddings,
            path = Config.Path.DATABASE_DIR,
            collection_name = Config.Database.DOCUMENT_COLLECTION
        )
        # Used BM25 for keyword search to be added for the Hybrid Search
        keyword_retriever = BM25Retriever.from_documents(
            documents = documents,
            k = Config.Retriever.K
        )
        return [vector_db, keyword_retriever]
