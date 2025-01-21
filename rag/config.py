"""The Configurations"""
import os
from pathlib import Path


class Config:
    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        DATABASE_DIR = APP_HOME / "docs-db"
        DOCUMENTS_DIR = APP_HOME / "tmp"

    class Database:
        DOCUMENT_COLLECTION = "documents"

    class Retriever:
        USE_RERANKER = True
        USE_CHAIN_FILTER = True
        K = 3

    DEBUG = True
    CONV_MESSAGES = 6
