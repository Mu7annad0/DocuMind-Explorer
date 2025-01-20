"""Retriever"""
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors.chain_filter import LLMChainFilter
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_cohere import CohereRerank

from rag.config import Config


def create_retriever(
        llm: BaseLanguageModel,
        vector_stores
) -> VectorStoreRetriever:
    """
    Creates an ensemble retriever.
    The ensemble retriever is a combination of two retrievers:
        1. Embedding-Based Retrieval: This is the vector database retriever.
        2. Term-Based Retrieval: This is the keyword retriever.
    The ensemble retriever is also known as the (hybrid search).

    The vector stores should contain two items:
        1. Vector Database: This will be used to create the embedding-based retrieval.
        2. Term-based retrieval: This is the keyword retriever.
    """
    vector_db, keyword_retriever = vector_stores
    retriever_vector_db = vector_db.as_retriever(
        retriever="similarity",
        search_kwargs={
            "k": Config.Retriever.K
        }
    )
    retriever = EnsembleRetriever(
        retrievers=[retriever_vector_db, keyword_retriever], weights=[0.5, 0.5]
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
            print(f"Error initializing CohereRerank: {e}")
            print("Falling back to retriever without reranking")

    if Config.Retriever.USE_CHAIN_FILTER:
        retriever = ContextualCompressionRetriever(
            base_compressor=LLMChainFilter.from_llm(llm),
            base_retriever=retriever
        )
    return retriever
