import os
from dataclasses import dataclass
from typing import Optional


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class RAGConfig:
    """Configuration for RAG system."""

    embedding_model: str = "BAAI/bge-large-en-v1.5"
    chunk_size: int = 700
    chunk_overlap: int = 100
    top_k: int = 3
    similarity_threshold: float = 0.0
    vector_db_path: str = os.path.join(BASE_DIR, "chroma_db")
    faq_json_path: str = os.path.join(BASE_DIR, "data", "tngd_faqs.json")
    use_reranker: bool = True
    groq_api_key: Optional[str] = None
