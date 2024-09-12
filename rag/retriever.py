from typing import Optional
import os

from langchain.retrievers import ContextualCompressionRetriever  # Import ContextualCompressionRetriever for advanced retrieval
from langchain.retrievers.document_compressors.chain_filter import LLMChainFilter  # Import LLMChainFilter for filtering retrieved documents
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank  # Import FlashrankRerank for reranking retrieved documents
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_qdrant import Qdrant

from rag.config import Config 

def create_retriever(
        llm: BaseLanguageModel, 
        vector_store: Optional[VectorStore] = None
) -> VectorStoreRetriever:  # Function returns a VectorStoreRetriever
    if vector_store is None:  # Check if a vector store was not provided
        vector_store = Qdrant.from_existing_collection(  # Create a new Qdrant vector store from an existing collection
            embedding=FastEmbedEmbeddings(model_name=Config.Model.EMBEDDING_MODEL),
            collection_name=Config.Database.DOCUMENT_COLLECTION,
            path=Config.Path.DATABASE_DIR
        )
    retriever = vector_store.as_retriever(  # Convert the vector store to a retriever
        search_type="similarity",  # Set the search type to similarity search
        search_kwargs={
            "k": Config.Retriever.K  # Set the number of results to retrieve from config
        }
    )
    if Config.Retriever.USE_RERANKER:  # Check if reranking is enabled in config
        # Set a specific cache directory
        os.environ['FLASHRANK_CACHE_DIR'] = os.path.expanduser('~/.cache/flashrank')
        
        try:
            reranker = FlashrankRerank(model=Config.Model.RERANKER)
            retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=retriever
            )
            print("Reranker initialized successfully")
        except Exception as e:
            print(f"Error initializing FlashrankRerank: {e}")
            print("Falling back to retriever without reranking")
            # Retriever remains unchanged if reranking fails

    if Config.Retriever.USE_CHAIN_FILTER:  # Check if chain filtering is enabled in config
        retriever = ContextualCompressionRetriever(  # Create a ContextualCompressionRetriever with chain filtering
            base_compressor=LLMChainFilter.from_llm(llm), base_retriever=retriever  # Initialize LLMChainFilter with the provided language model
        )

    return retriever