import os
from dataclasses import dataclass


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class RAGConfig:
    """Configuration for RAG system."""

    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    chunk_size: int = 700
    chunk_overlap: int = 100
    top_k: int = 3
    similarity_threshold: float = 0.0
    llm_model: str = "microsoft/Phi-3-mini-4k-instruct"
    vector_db_path: str = os.path.join(BASE_DIR, "chroma_db")
    faq_json_path: str = os.path.join(BASE_DIR, "data", "tngd_faqs.json")
    use_reranker: bool = True
