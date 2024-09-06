from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFium2Loader  # Import PDF loader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings  # Import embedding model
from langchain_core.vectorstores import VectorStore  # Import VectorStore type
from langchain_experimental.text_splitter import SemanticChunker  # Import semantic text splitter
from langchain_qdrant import Qdrant  # Import Qdrant vector store
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Import recursive text splitter

from rag.config import Config


# Here's How the Ingestor works:

# 1. File Handling: It takes PDF files that users upload to our system.

# 2. Text Extraction: It converts these PDFs into plain text, making them easier to work with.

# 3. Chunking: The Ingestor then breaks down this text into smaller, manageable pieces (or "chunks"). 

# 4. Embedding: For each chunk, it creates a special numerical representation called an "embedding". 

# 5. Storage: Finally, it saves these text chunks and their embeddings in a special type of database 
#    called a Vector database. This database is optimized for quickly finding similar pieces of text.

# By doing all this, the Ingestor prepares our documents in a way that makes it easy and fast to 
# search through them later, even when we have a large number of documents.


class Ingestor:
    def __init__(self):
        self.embeddings = FastEmbedEmbeddings(model_name=Config.Model.EMBEDDING_MODEL)  # Initialize embedding model
        self.semantic_splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="interquartile"
        )  # Initialize semantic text splitter
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 2048,
            chunk_overlap = 100,
            add_start_index=True
        )  # Initialize recursive text splitter

    def ingest(self, doc_files: List[Path]) -> VectorStore:
        documents = []  # Initialize empty list to store processed documents
        for doc_file in doc_files:  # Iterate through each document file
            loaded_docs = PyPDFium2Loader(doc_file).load()  # Load PDF document
            document_text = "\n".join([doc.page_content for doc in loaded_docs])  # Combine all pages into single text
            documents.extend(self.recursive_splitter.split_documents(
                self.semantic_splitter.create_documents([document_text])
            ))  # Split document semantically and add to documents list
        return Qdrant.from_documents(  # Create and return Qdrant vector store
            documents = documents,  # Pass processed documents
            embedding=self.embeddings,  # Pass embedding model
            path = Config.Path.DATABASE_DIR,  # Set database directory path
            collection_name = Config.Database.DOCUMENT_COLLECTION  # Set collection name
        )