import os
from pathlib import Path

class Config:
    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        DATABASE_DIR = APP_HOME / "docs-db"
        DOCUMENTS_DIR = APP_HOME / "tmp"
        IMAGES_DIR = APP_HOME / "images"

    class Database:
        DOCUMENT_COLLECTION = "documents"

    class Model:
        EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
        RERANKER = "ms-marco-MiniLM-L-12-v2"
        LOCAL_LLM = "gemma2:9b"
        TEMP = 0.0
        MAX_TOKENS = 7000
        USE_LOCAL_LLM = True

    class Retriever:
        USE_RERANKER = True
        USE_CHAIN_FILTER = False
        K = 5

    DEBUG = True
    CONV_MESSAGES = 6
