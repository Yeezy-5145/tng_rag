"""
RAG System for TNG Digital FAQ
Complete RAG pipeline with retrieval, generation, and guardrails
"""

import json
import os
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


@dataclass
class RAGConfig:
    """Configuration for RAG system"""

    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 3
    similarity_threshold: float = 0.3
    llm_model: str = "microsoft/DialoGPT-medium"
    vector_db_path: str = "./chroma_db"
    faq_json_path: str = "./tngd_faqs.json"


class DocumentChunker:
    """Handles document chunking with overlap"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text with overlap strategy.

        Strategy:
        - Split by sentences first to preserve semantic meaning
        - Use sliding window with overlap to maintain context
        - Ensure chunks don't break mid-sentence
        """
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())

            # If adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap
                # Keep last N sentences for overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    s_len = len(s.split())
                    if overlap_length + s_len <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += s_len
                    else:
                        break

                current_chunk = overlap_sentences + [sentence]
                current_length = overlap_length + sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks if chunks else [text]


class AdversarialGuardrails:
    """Guardrails against adversarial prompts"""

    # Default message when no answer is found
    NO_ANSWER_MESSAGE = (
        "I don't have information in my FAQ knowledge base to answer that."
    )

    # Patterns that indicate prompt injection attempts
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|all|above)",
        r"forget\s+(everything|all|previous)",
        r"you\s+are\s+(now|a)",
        r"act\s+as\s+if",
        r"pretend\s+to\s+be",
        r"system\s*:",
        r"<\|.*?\|>",
        r"\[INST\]",
        r"###\s*(instruction|system|prompt)",
        r"disregard\s+(all|previous|above)",
    ]

    # Suspicious keywords
    SUSPICIOUS_KEYWORDS = [
        "hack",
        "exploit",
        "bypass",
        "jailbreak",
        "override",
        "admin",
        "root",
        "sudo",
        "password",
        "token",
        "api key",
    ]

    def __init__(self):
        self.injection_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]

    def detect_injection(self, text: str) -> bool:
        """Detect potential prompt injection attempts"""
        text_lower = text.lower()

        # Check for injection patterns
        for pattern in self.injection_patterns:
            if pattern.search(text):
                return True

        # Check for suspicious keywords (context-dependent)
        suspicious_count = sum(1 for kw in self.SUSPICIOUS_KEYWORDS if kw in text_lower)
        if suspicious_count >= 2:  # Multiple suspicious keywords
            return True

        return False

    def sanitize_input(self, text: str) -> str:
        """Sanitize user input"""
        # Remove potential control characters
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
        # Limit length
        if len(text) > 1000:
            text = text[:1000]
        return text.strip()

    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate user query.
        Returns (is_valid, error_message)
        """
        if not query or len(query.strip()) < 3:
            return False, "Query too short. Please provide a more detailed question."

        if len(query) > 1000:
            return (
                False,
                "Query too long. Please keep your question under 1000 characters.",
            )

        if self.detect_injection(query):
            return (
                False,
                "Invalid query detected. Please ask a legitimate question about TNG Digital services.",
            )

        return True, None


class RAGSystem:
    """Main RAG system class"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.chunker = DocumentChunker(config.chunk_size, config.chunk_overlap)
        self.guardrails = AdversarialGuardrails()

        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(config.embedding_model)

        # Initialize vector store
        print("Initializing vector store...")
        self.client = chromadb.PersistentClient(
            path=config.vector_db_path, settings=Settings(anonymized_telemetry=False)
        )
        self.collection = None

        # Initialize LLM
        self.llm = None
        self.tokenizer = None
        self._init_llm()

    def _init_llm(self):
        """Initialize LLM for generation"""
        print("Loading LLM model (this may take a while)...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model)
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model,
            torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
            device_map="auto" if torch.cuda.is_available() else None,
        )
        print("LLM model loaded successfully")

    def build_knowledge_base(self, faqs: List[Dict]):
        """Build knowledge base from FAQs"""
        print("Building knowledge base...")

        # Create or get collection
        try:
            self.collection = self.client.get_collection("tngd_faqs")
            print("Using existing collection")
        except Exception:
            self.collection = self.client.create_collection(
                name="tngd_faqs",
                metadata={"description": "TNG Digital FAQ knowledge base"},
            )
            print("Created new collection")

        # Check if collection is empty
        if self.collection.count() == 0:
            print("Indexing documents...")
            documents = []
            metadatas = []
            ids = []

            for faq in faqs:
                # Chunk the answer
                chunks = self.chunker.chunk_text(faq.get("answer", ""))

                for i, chunk in enumerate(chunks):
                    doc_id = f"{faq.get('id', 'unknown')}_chunk_{i}"
                    documents.append(chunk)
                    metadatas.append(
                        {
                            "question": faq.get("question", ""),
                            "url": faq.get("url", ""),
                            "category": faq.get("category", "General"),
                            "chunk_index": i,
                            "faq_id": faq.get("id", "unknown"),
                        }
                    )
                    ids.append(doc_id)

            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i : i + batch_size]
                batch_metas = metadatas[i : i + batch_size]
                batch_ids = ids[i : i + batch_size]

                self.collection.add(
                    documents=batch_docs, metadatas=batch_metas, ids=batch_ids
                )
                print(
                    f"Indexed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}"
                )

            print(
                f"Knowledge base built with {len(documents)} chunks from {len(faqs)} FAQs"
            )
        else:
            print(f"Knowledge base already contains {self.collection.count()} chunks")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve relevant documents"""
        if top_k is None:
            top_k = self.config.top_k

        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query], convert_to_numpy=True
        ).tolist()

        # Query the collection
        try:
            # Try with query_texts first (ChromaDB auto-embeds)
            results = self.collection.query(query_texts=[query], n_results=top_k)
        except Exception:
            # Fallback to query_embeddings if query_texts doesn't work
            results = self.collection.query(
                query_embeddings=query_embedding, n_results=top_k
            )

        retrieved_docs = []
        if results.get("documents") and len(results["documents"][0]) > 0:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]

                # Handle distance/similarity
                if "distances" in results and results["distances"]:
                    distance = results["distances"][0][i]
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity = 1 - distance if distance is not None else 1.0
                else:
                    similarity = 1.0  # Default if no distance provided

                if similarity >= self.config.similarity_threshold:
                    retrieved_docs.append(
                        {
                            "text": doc,
                            "question": metadata.get("question", ""),
                            "url": metadata.get("url", ""),
                            "category": metadata.get("category", "General"),
                            "similarity": similarity,
                        }
                    )

        return retrieved_docs

    def generate_response(self, query: str, context_docs: List[Dict]) -> str:
        """Generate response using LLM"""
        if not context_docs:
            return AdversarialGuardrails.NO_ANSWER_MESSAGE

        # Build context from retrieved documents
        context = "\n\n".join(
            [f"Q: {doc['question']}\nA: {doc['text']}" for doc in context_docs]
        )

        # Create prompt
        prompt = f"""Based on the following TNG Digital FAQ information, answer the user's question accurately and concisely. Only use information from the provided context. If the context doesn't contain enough information, say so.

Context:
{context}

User Question: {query}

Answer:"""

        # Generate response using transformers
        inputs = self.tokenizer.encode(
            prompt, return_tensors="pt", max_length=1024, truncation=True
        )
        with torch.no_grad():
            outputs = self.llm.generate(
                inputs,
                max_length=inputs.shape[1] + 200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(
            outputs[0][inputs.shape[1] :], skip_special_tokens=True
        )
        return response.strip()

    def query(self, user_query: str) -> Dict:
        """
        Main query interface for the RAG system.

        Args:
            user_query: User's question

        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Validate and sanitize input
        is_valid, error_msg = self.guardrails.validate_query(user_query)
        if not is_valid:
            return {"answer": error_msg, "sources": [], "error": True}

        sanitized_query = self.guardrails.sanitize_input(user_query)

        # Retrieve relevant documents
        retrieved_docs = self.retrieve(sanitized_query)

        if not retrieved_docs:
            return {
                "answer": AdversarialGuardrails.NO_ANSWER_MESSAGE,
                "sources": [],
                "error": False,
            }

        # Generate response
        answer = self.generate_response(sanitized_query, retrieved_docs)

        # Ensure answer only uses verified FAQ information
        # Additional validation: check if answer contains suspicious content
        if self.guardrails.detect_injection(answer):
            # Fallback to most relevant document
            answer = retrieved_docs[0]["text"]

        return {
            "answer": answer,
            "sources": [
                {
                    "question": doc["question"],
                    "url": doc["url"],
                    "category": doc["category"],
                    "similarity": doc["similarity"],
                }
                for doc in retrieved_docs
            ],
            "error": False,
        }


def load_faqs(json_path: str) -> List[Dict]:
    """Load FAQs from JSON file"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def initialize_rag_system(config: RAGConfig = None) -> RAGSystem:
    """Initialize RAG system with knowledge base"""
    if config is None:
        config = RAGConfig()

    rag = RAGSystem(config)

    # Load FAQs
    if os.path.exists(config.faq_json_path):
        faqs = load_faqs(config.faq_json_path)
        rag.build_knowledge_base(faqs)
    else:
        print(f"Warning: FAQ file not found at {config.faq_json_path}")
        print("Please run scraper.py first to collect FAQ data.")

    return rag


# Main function interface
def ask_tngd_bot(question: str) -> dict:
    """
    Accepts a question string and returns answer with retrieved context.

    Args:
        question: User's question about TNG Digital

    Returns:
        dict: {
            "question": str,
            "retrieved_chunks": list,
            "final_answer": str,
            "blocked": bool
        }
    """
    # Initialize RAG system (cached after first call)
    if not hasattr(ask_tngd_bot, "_rag_system"):
        try:
            ask_tngd_bot._rag_system = initialize_rag_system()
        except Exception as e:
            return {
                "question": question,
                "retrieved_chunks": [],
                "final_answer": AdversarialGuardrails.NO_ANSWER_MESSAGE,
                "blocked": False,
            }

    rag = ask_tngd_bot._rag_system

    # Check if RAG system is properly initialized
    if rag is None or rag.collection is None or rag.collection.count() == 0:
        return {
            "question": question,
            "retrieved_chunks": [],
            "final_answer": AdversarialGuardrails.NO_ANSWER_MESSAGE,
            "blocked": False,
        }

    # Validate query with guardrails
    is_valid, error_msg = rag.guardrails.validate_query(question)

    if not is_valid:
        return {
            "question": question,
            "retrieved_chunks": [],
            "final_answer": error_msg,
            "blocked": True,
        }

    # Sanitize input
    sanitized_query = rag.guardrails.sanitize_input(question)

    # Retrieve relevant documents
    retrieved_docs = rag.retrieve(sanitized_query)

    if not retrieved_docs:
        return {
            "question": question,
            "retrieved_chunks": [],
            "final_answer": AdversarialGuardrails.NO_ANSWER_MESSAGE,
            "blocked": False,
        }

    # Format retrieved chunks
    retrieved_chunks = [
        {
            "text": doc["text"],
            "question": doc["question"],
            "url": doc["url"],
            "category": doc["category"],
            "similarity": doc["similarity"],
        }
        for doc in retrieved_docs
    ]

    # Generate response
    final_answer = rag.generate_response(sanitized_query, retrieved_docs)

    # Validate response doesn't contain injection
    if rag.guardrails.detect_injection(final_answer):
        # Fallback to most relevant document
        final_answer = retrieved_docs[0]["text"]

    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "final_answer": final_answer,
        "blocked": False,
    }


# Legacy function interface (for backward compatibility)
def answer_question(question: str, rag_system: RAGSystem = None) -> Dict:
    """
    Legacy function interface for answering questions.

    Args:
        question: User's question about TNG Digital
        rag_system: Optional pre-initialized RAG system (for efficiency)

    Returns:
        Dictionary with answer and sources
    """
    if rag_system is None:
        rag_system = initialize_rag_system()

    return rag_system.query(question)


if __name__ == "__main__":
    # Example usage
    print("Initializing RAG system...")
    rag = initialize_rag_system()

    # Test queries
    test_queries = [
        "How do I top up my TNG Digital wallet?",
        "What is TNG eWallet?",
        "How do I reset my password?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("-" * 60)
        result = rag.query(query)
        print(f"Answer: {result['answer']}")
        if result["sources"]:
            print(f"\nSources:")
            for src in result["sources"]:
                print(f"  - {src['question']} ({src['url']})")
