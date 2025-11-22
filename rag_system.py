"""
RAG System for TNG Digital FAQ
Complete RAG pipeline with retrieval, generation, and guardrails
"""

import json
import os
import re
import hashlib
from typing import List, Dict, Optional, Tuple
import numpy as np

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from rag_config import RAGConfig
from rag_chunker import DocumentChunker
from rag_guardrails import AdversarialGuardrails


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

    def _call_llm(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        do_sample: bool = True,
    ) -> str:
        """
        Central helper for all LLM generation calls.
        Ensures consistent tokenization, truncation, and decoding.
        """
        inputs = self.tokenizer.encode(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        input_length = inputs.shape[1]

        with torch.no_grad():
            outputs = self.llm.generate(
                inputs,
                max_length=input_length + max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=0.95,
                top_k=40,
                repetition_penalty=1.2,
                no_repeat_ngram_size=4,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()

        return response

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
                metadata={
                    "description": "TNG Digital FAQ knowledge base",
                    "version": "2.0",  # Version for format tracking
                },
            )
            print("Created new collection")

        # Check if collection is empty or needs rebuilding
        # Rebuild if collection has old format (check for 'unknown' IDs or missing version)
        needs_rebuild = False
        collection_count = 0
        try:
            collection_count = self.collection.count()
            if collection_count > 0:
                # Check collection metadata for version
                collection_meta = self.collection.metadata or {}
                collection_version = collection_meta.get("version", "1.0")

                # Sample a few documents to check format
                try:
                    sample = self.collection.get(limit=min(10, collection_count))
                    if sample and sample.get("ids"):
                        # Check if IDs contain "unknown" which indicates old format
                        sample_ids = " ".join(
                            sample["ids"][: min(10, len(sample["ids"]))]
                        )
                        # Check if metadata has answer_text field (new format indicator)
                        sample_metadata = sample.get("metadatas", [])
                        has_answer_text = False
                        if sample_metadata:
                            for meta in sample_metadata[:5]:
                                if "answer_text" in meta:
                                    has_answer_text = True
                                    break

                        # Rebuild if old format detected or version mismatch
                        if (
                            "unknown" in sample_ids.lower()
                            or collection_version != "2.0"
                            or not has_answer_text
                        ):
                            print("Detected old format - rebuilding knowledge base...")
                            needs_rebuild = True
                            # Delete old collection
                            try:
                                self.client.delete_collection("tngd_faqs")
                            except Exception as e:
                                print(f"Warning: Could not delete old collection: {e}")
                            self.collection = self.client.create_collection(
                                name="tngd_faqs",
                                metadata={
                                    "description": "TNG Digital FAQ knowledge base",
                                    "version": "2.0",
                                },
                            )
                except Exception as e:
                    # If we can't sample, rebuild anyway
                    print(f"Error checking collection format ({e}) - rebuilding...")
                    needs_rebuild = True
                    try:
                        self.client.delete_collection("tngd_faqs")
                    except Exception:
                        pass
                    self.collection = self.client.create_collection(
                        name="tngd_faqs",
                        metadata={
                            "description": "TNG Digital FAQ knowledge base",
                            "version": "2.0",
                        },
                    )
        except Exception as e:
            print(f"Error checking collection ({e}) - will rebuild")
            needs_rebuild = True

        # Check if collection is empty or needs rebuilding
        if collection_count == 0 or needs_rebuild:
            print("Indexing documents...")
            documents = []
            metadatas = []
            ids = []

            for idx, faq in enumerate(faqs):
                # Chunk the answer
                answer_text = faq.get("answer", "")
                chunks = self.chunker.chunk_text(answer_text) if answer_text else []

                # Get category - handle both "category" (string) and "categories" (list)
                category = "General"
                if "categories" in faq and faq["categories"]:
                    if isinstance(faq["categories"], list):
                        category = (
                            faq["categories"][0] if faq["categories"] else "General"
                        )
                    else:
                        category = faq["categories"]
                elif "category" in faq:
                    category = faq["category"]

                # Generate unique ID from question hash or URL
                question = faq.get("question", "")
                url = faq.get("url", "")
                unique_key = url if url else question
                faq_hash = hashlib.md5(unique_key.encode()).hexdigest()[:8]

                # Index question + answer together for better matching
                # This helps match queries to questions
                question_plus_answer = f"{question}\n{answer_text}".strip()

                # If no chunks, create one from the full text
                if not chunks:
                    if question_plus_answer:
                        chunks = [question_plus_answer]
                    elif answer_text:
                        chunks = [answer_text]
                    else:
                        continue  # Skip if no content

                # Add question as first chunk for better question matching
                # This allows queries to match against questions directly
                for i, chunk in enumerate(chunks):
                    # For first chunk, include question in the document text for better matching
                    if i == 0 and question:
                        # Combine question and answer in first chunk for semantic search
                        chunk_text = f"{question} {chunk}"
                    else:
                        chunk_text = chunk

                    doc_id = f"{faq_hash}_chunk_{i}"
                    documents.append(chunk_text)
                    metadatas.append(
                        {
                            "question": question,
                            "url": url,
                            "category": category,
                            "chunk_index": i,
                            "faq_index": idx,
                            "answer_text": (
                                answer_text[:500] if answer_text else ""
                            ),  # Store answer for fallback
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

    def retrieve(
        self, query: str, top_k: int = None, debug: bool = False
    ) -> List[Dict]:
        """
        Retrieve top-K most relevant documents using:
        1. Embedding similarity search (broad recall)
        2. Optional LLM reranking (precision)
        3. Deduplication + diversity filtering
        """

        if top_k is None:
            top_k = self.config.top_k

        # 1) If the collection is empty, return nothing
        if not self.collection or self.collection.count() == 0:
            if debug:
                print("DEBUG: Collection empty.")
            return []

        # 2) Do a wide search to ensure recall (higher than final K)
        wide_k = min(max(top_k * 8, 20), self.collection.count())  # Recall > precision

        try:
            results = self.collection.query(query_texts=[query], n_results=wide_k)
        except:
            query_embedding = self.embedding_model.encode(
                [query], convert_to_numpy=True
            ).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding, n_results=wide_k
            )

        # 3) Convert raw Chroma result format → internal objects
        docs: List[Dict] = []
        query_lower = query.lower()
        for i, text in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]

            distance = results.get("distances", [[None]])[0][i]
            semantic_similarity = (
                max(0.0, 1.0 - distance) if distance is not None else 0.0
            )

            # Normalise metadata so multiple RAG data objects can be merged cleanly
            categories_meta = meta.get("categories")
            if isinstance(categories_meta, list):
                categories = categories_meta
            else:
                single_category = meta.get("category")
                categories = [single_category] if single_category else []

            question_text = (meta.get("question") or "").strip()
            question_lower = question_text.lower()

            # Lightweight lexical bonus so exact/near-exact question titles win
            lexical_bonus = 0.0
            if question_lower:
                if question_lower == query_lower:
                    lexical_bonus = 0.5
                elif question_lower in query_lower or query_lower in question_lower:
                    lexical_bonus = 0.3

            docs.append(
                {
                    "text": meta.get("answer_text", text),
                    "question": question_text,
                    "url": meta.get("url", ""),
                    "categories": categories,
                    "similarity": float(semantic_similarity + lexical_bonus),
                }
            )

        # 4) Remove near-duplicate answers (same text, different metadata)
        seen = set()
        unique_docs = []
        for d in docs:
            key = d["text"].strip().lower()
            if key not in seen:
                seen.add(key)
                unique_docs.append(d)

        docs = unique_docs

        # 5) Filter out low-relevance items with a threshold
        threshold = getattr(self.config, "similarity_threshold", 0.25)
        docs = [d for d in docs if d["similarity"] >= threshold]

        if debug:
            print(f"DEBUG: After threshold filter: {len(docs)} docs")

        if not docs:
            return []

        # ===== Optional Step: LLM-based Reranking (hugely improves retrieval) =====
        if getattr(self.config, "use_reranker", False):
            try:
                # Prepare a compact text prompt for the causal LM
                doc_snippets = []
                for i, d in enumerate(docs):
                    # Include both FAQ question and answer text so the reranker sees intent + content
                    snippet = f"Q: {d['question']}\nA: {d['text']}"
                    if len(snippet) > 300:
                        snippet = snippet[:300] + "..."
                    doc_snippets.append(f"[{i}] {snippet}")

                ranking_prompt = (
                    "You are helping to rank FAQ answers by how relevant they are to a user question.\n\n"
                    f"Question: {query}\n\n"
                    "Here are candidate answers:\n" + "\n".join(doc_snippets) + "\n\n"
                    "Return ONLY a JSON array of the document indices in order of relevance, "
                    "for example: [2, 0, 1]. Do not add any other text.\n"
                )

                response_text = self._call_llm(
                    ranking_prompt, max_new_tokens=128, temperature=0.3, do_sample=False
                )

                # Try to parse JSON; if that fails, fall back to regex of integers
                ranked_indices = []
                try:
                    ranked_indices = json.loads(response_text)
                except Exception:
                    idx_strings = re.findall(r"\d+", response_text)
                    seen_idx = set()
                    for s in idx_strings:
                        idx = int(s)
                        if idx not in seen_idx and 0 <= idx < len(docs):
                            seen_idx.add(idx)
                            ranked_indices.append(idx)

                if ranked_indices:
                    # Preserve reranker ordering exactly; don't re-sort by similarity afterwards
                    docs = [docs[i] for i in ranked_indices if 0 <= i < len(docs)]

            except Exception as e:
                if debug:
                    print(
                        f"DEBUG: Reranker failed with error {e}, skipping LLM reranking."
                    )

        # 6) Final sort & top_k cut
        # If reranker was used, docs are already ordered by LLM relevance.
        # Otherwise, fall back to similarity-based ordering.
        if not getattr(self.config, "use_reranker", False):
            docs = sorted(docs, key=lambda d: d["similarity"], reverse=True)

        # 7) Return final top-K
        final_docs = docs[:top_k]

        if debug:
            print(f"DEBUG: Final retrieved documents:")
            for i, d in enumerate(final_docs):
                print(f"  {i+1}. sim={d['similarity']:.3f} | {d['question'][:80]}...")

        return final_docs

    def _post_process_response(self, response: str, sources: List[Dict]) -> str:
        """
        Post-process the model output by repairing incomplete words,
        restoring cut-off sentences, removing odd formatting, and
        improving natural flow.

        This uses a second LLM pass rather than manual regex heuristics.
        """

        if not response:
            return response

        # === 1. Clean obvious formatting noise (optional but helpful) ===
        # Remove double spaces, fix spacing around punctuation.
        response = re.sub(r"\s+", " ", response)
        response = re.sub(r"\s+([.,!?])", r"\1", response)
        response = response.strip()

        # Strip obvious boilerplate phrases from FAQ content
        boilerplate_patterns = [
            r"Below are related articles that might be useful for you.*",
            r"For more information, you may read the Product Disclosure Sheet here.*",
        ]
        for pat in boilerplate_patterns:
            response = re.sub(pat, "", response, flags=re.IGNORECASE)

        return response

    def _build_retrieved_context(self, context_docs: List[Dict]) -> str:
        """
        Build a structured multi-chunk context block for LLM synthesis.
        Each chunk is clearly labeled, includes metadata, and avoids
        implying it is the only correct answer.
        """
        parts: List[str] = []
        for i, doc in enumerate(context_docs):
            question = doc.get("question", "").strip()
            text = doc.get("text", "").strip()
            source = doc.get("url") or ""
            categories = ", ".join(doc.get("categories") or [])

            chunk_lines = [f"--- Chunk {i+1} ---"]
            if question:
                chunk_lines.append(f"Question: {question}")
            if categories:
                chunk_lines.append(f"Category: {categories}")
            if source:
                chunk_lines.append(f"Source: {source}")
            if text:
                chunk_lines.append(f"Content: {text}")

            parts.append("\n".join(chunk_lines))

        return "\n\n".join(parts)

    def generate_response(self, query: str, context_docs: List[Dict]) -> str:
        """
        Generate a synthesized, paraphrased response using multiple retrieved chunks.
        Explicitly instructs the LLM to combine multiple sources.
        """

        if not context_docs:
            return (
                "Hmm, I don’t see anything in my knowledge base that directly answers this question. "
                "I can explain general TNG Digital services if you like."
            )

        # Normalize and sort by similarity
        for doc in context_docs:
            doc["text"] = doc.get("text") or doc.get("answer") or ""
            doc["similarity"] = float(doc.get("similarity", 0.0))

        sorted_docs = sorted(context_docs, key=lambda d: d["similarity"], reverse=True)
        # Use fewer chunks in the prompt to keep answers focused and natural
        max_docs_for_prompt = min(len(sorted_docs), 4)
        selected_docs = sorted_docs[:max_docs_for_prompt]

        retrieved_context = self._build_retrieved_context(selected_docs)

        # Strong, explicit prompt for multi-chunk synthesis
        persona_prompt = f"""
You are a friendly, helpful AI assistant. Your answers should be clear, concise, and human-like.

INSTRUCTIONS:
- Review ALL provided chunks carefully.
- Combine relevant information from multiple chunks into a single answer.
- Paraphrase in your own words; do NOT copy sentences verbatim.
- Only use the information provided in the chunks.
- Prioritize chunks whose question/title best matches the user's question.
- Ignore chunks that are irrelevant to the question.
- Do not mention FAQ article titles or say things like "Below are related articles".
- Aim for 2–4 sentences unless the user explicitly asks for more detail.
- Maintain a friendly, conversational tone.

OUTPUT STRUCTURE:
Answer:
<direct answer here>

Explanation:
<short explanation, based on the context>

Next steps:
<helpful next steps or "None needed.">

User question:
{query}

Retrieved Context:
{retrieved_context}
"""

        try:
            response = self._call_llm(
                persona_prompt,
                max_new_tokens=512,
                temperature=0.3,  # keep low for factuality
                do_sample=True,
            )

            # Ensure proper punctuation
            if response and response[-1] not in ".!?":
                response += "."

            # Fallback if output is too short
            if not response or len(response) < 20:
                fallback = " ".join(
                    d["text"] for d in selected_docs[:2]
                )  # combine top 2 chunks
                if fallback and fallback[-1] not in ".!?":
                    fallback += "."
                response = fallback

        except Exception:
            # On error, fallback to combined top 2 chunks
            fallback = " ".join(d["text"] for d in selected_docs[:2])
            if fallback and fallback[-1] not in ".!?":
                fallback += "."
            response = fallback

        return response

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
        retrieved_docs = self.retrieve(sanitized_query, debug=False)

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
                    "category": _extract_primary_category(doc),
                    "similarity": doc["similarity"],
                }
                for doc in retrieved_docs
            ],
            "error": False,
        }


def _parse_structured_answer(text: str) -> Dict[str, str]:
    """
    Parse the structured persona output into sections:
    Answer, Explanation, Next steps.

    This reduces manual checking for downstream consumers while still
    keeping the full text available.
    """
    sections = {"answer": "", "explanation": "", "next_steps": ""}

    if not text:
        return sections

    current = None
    lines = text.splitlines()

    for raw_line in lines:
        line = raw_line.strip()
        lower = line.lower()

        if lower.startswith("answer:"):
            current = "answer"
            # Capture any content on the same line after "Answer:"
            remainder = line[len("answer:") :].strip()
            if remainder:
                sections["answer"] += remainder + " "
            continue

        if lower.startswith("explanation:"):
            current = "explanation"
            remainder = line[len("explanation:") :].strip()
            if remainder:
                sections["explanation"] += remainder + " "
            continue

        if lower.startswith("next steps:") or lower.startswith("next step:"):
            current = "next_steps"
            remainder = line.split(":", 1)[1].strip() if ":" in line else ""
            if remainder:
                sections["next_steps"] += remainder + " "
            continue

        if current in sections and line:
            sections[current] += line + " "

    # Final cleanup: strip extra whitespace
    for key in sections:
        sections[key] = sections[key].strip()

    return sections


def _extract_primary_category(doc: Dict) -> str:
    """
    Safely extract a single category string from a document that may contain
    either a 'category' field or a 'categories' list. This prevents KeyError
    when different RAG sources use different metadata shapes.
    """
    # Explicit single category wins if present
    if "category" in doc and doc["category"]:
        return str(doc["category"])

    # Fallback to first element of categories list
    categories = doc.get("categories")
    if isinstance(categories, list) and categories:
        return str(categories[0])
    if isinstance(categories, str) and categories:
        return categories

    return ""


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

    # Retrieve relevant documents (enable debug for troubleshooting)
    retrieved_docs = rag.retrieve(sanitized_query, debug=False)

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
            "category": _extract_primary_category(doc),
            "similarity": doc["similarity"],
        }
        for doc in retrieved_docs
    ]

    # Log RAG state before synthesis for easier debugging/inspection
    try:
        debug_snapshot = {
            "question": question,
            "retrieved_chunks": retrieved_chunks,
            "final_answer": "",
            "blocked": False,
        }
        print("\n[DEBUG] RAG state before synthesis:")
        print(json.dumps(debug_snapshot, ensure_ascii=False, indent=2))
    except Exception:
        # Logging should never break the main flow
        pass

    # Generate response using persona logic only (retrieval is already done)
    final_answer = rag.generate_response(sanitized_query, retrieved_docs)

    # Validate response doesn't contain injection
    if rag.guardrails.detect_injection(final_answer):
        # Fallback to most relevant document
        final_answer = retrieved_docs[0]["text"]

    # Parse structured output into sections for easier consumption
    parsed = _parse_structured_answer(final_answer)

    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "final_answer": final_answer,
        "final_answer_parsed": parsed,
        "blocked": False,
    }
