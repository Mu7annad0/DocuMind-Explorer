"""Retriever"""
from typing import Optional

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_filter import LLMChainFilter
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_qdrant import Qdrant
from langchain_cohere import CohereEmbeddings, CohereRerank

from rag.config import Config


def create_retriever(
        llm: BaseLanguageModel,
        vector_store: Optional[VectorStore] = None
) -> VectorStoreRetriever:
    """Crearte the retrievar"""
    if vector_store is None:
        vector_store = Qdrant.from_existing_collection(
            collection_name=Config.Database.DOCUMENT_COLLECTION,
            path=Config.Path.DATABASE_DIR,
            embedding=CohereEmbeddings(model="embed-english-v3.0")
        )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": Config.Retriever.K
        }
    )
    if Config.Retriever.USE_RERANKER:
        try:
            reranker = CohereRerank(model="rerank-v3.5")
            retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=retriever
            )
            print("Reranker initialized successfully")
        except (ValueError, TypeError) as e:
            print(f"Error initializing FlashrankRerank: {e}")
            print("Falling back to retriever without reranking")

    if Config.Retriever.USE_CHAIN_FILTER:
        retriever = ContextualCompressionRetriever(
            base_compressor=LLMChainFilter.from_llm(llm),
            base_retriever=retriever
        )
    return retriever
