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
        self._last_best_chunk: Optional[Dict] = None  # Track last chunk used for answer

        # Determine device (GPU if available, else CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")

        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(config.embedding_model)
        # Move embedding model to GPU if available
        if self.device == "cuda":
            self.embedding_model = self.embedding_model.to(self.device)
            print("Embedding model moved to GPU")

        # Initialize vector store
        print("Initializing vector store...")
        self.client = chromadb.PersistentClient(
            path=config.vector_db_path, settings=Settings(anonymized_telemetry=False)
        )
        self.collection = None

        # Initialize LLM
        self.llm = None
        self.tokenizer = None
        self.device_map = None  # Track if using device_map="auto"
        self._init_llm()

    def _init_llm(self):
        """Initialize LLM for generation"""
        print("Loading LLM model (this may take a while)...")
        try:
            # Try to load with fast tokenizer first, fall back to slow if sentencepiece not available
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model)
            except ValueError as e:
                if "sentencepiece" in str(e).lower():
                    print("⚠ Warning: Fast tokenizer requires sentencepiece. Using slow tokenizer instead.")
                    print("   Install sentencepiece with: pip install sentencepiece")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model, use_fast=False)
                else:
                    raise
            
            # Set pad token if not present (required for some models)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure model loading for GPU
            if self.device == "cuda":
                try:
                    self.llm = AutoModelForCausalLM.from_pretrained(
                        self.config.llm_model,
                        dtype=torch.float16,  # Use float16 for GPU to save memory (fixed deprecation)
                        device_map="auto",  # Automatically distribute across GPUs
                    )
                    self.device_map = "auto"  # Track that we're using device_map
                    print("✓ LLM model loaded on GPU")
                except Exception as e:
                    print(f"⚠ Warning: Failed to load LLM on GPU: {e}")
                    print("Falling back to CPU...")
                    self.llm = AutoModelForCausalLM.from_pretrained(
                        self.config.llm_model,
                        dtype=torch.float32,
                        device_map=None,
                    )
                    self.device_map = None
                    self.llm = self.llm.to("cpu")
                    self.device = "cpu"  # Update device to reflect actual usage
                    print("✓ LLM model loaded on CPU (fallback)")
            else:
                self.llm = AutoModelForCausalLM.from_pretrained(
                    self.config.llm_model,
                    dtype=torch.float32,
                    device_map=None,
                )
                self.device_map = None  # Not using device_map
                # Explicitly move to CPU if not using device_map
                self.llm = self.llm.to(self.device)
                print("✓ LLM model loaded on CPU")
            
            # Verify model is loaded
            if self.llm is None:
                raise RuntimeError("LLM model failed to initialize")
            
            # Check which device the model is actually on
            if hasattr(self.llm, 'device'):
                actual_device = next(self.llm.parameters()).device
                print(f"✓ LLM model verified on device: {actual_device}")
            elif self.device_map == "auto":
                print("✓ LLM model using device_map='auto' (distributed across available devices)")
            
            print("✓ LLM model loaded successfully")
        except Exception as e:
            print(f"✗ ERROR: Failed to load LLM model: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise

    def _call_llm(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = True,
    ) -> str:
        """
        Central helper for all LLM generation calls.
        Ensures consistent tokenization, truncation, and decoding.
        """
        # Check if tokenizer has a chat template (for instruction-tuned models like LLaMA 3.1)
        # If it does, format the prompt using the chat template
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            # Format as a single user message for instruction following
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        
        # Tokenize the prompt
        # Llama-3.1-8B-Instruct supports up to 128k tokens, but we'll use 8192 for safety
        inputs = self.tokenizer.encode(
            formatted_prompt, return_tensors="pt", truncation=True, max_length=8192
        )
        
        # Move inputs to the same device as the model
        # If using device_map="auto", find the device of the model's input layer
        if self.device_map == "auto" and self.llm is not None:
            # With device_map="auto", check where the model's first layer is
            try:
                model_device = next(self.llm.parameters()).device
                inputs = inputs.to(model_device)
            except Exception:
                # Fallback: move to cuda if available
                inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
        elif self.device_map is None:
            inputs = inputs.to(self.device)
        else:
            # Fallback to device setting
            inputs = inputs.to(self.device)

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
                # Ensure the model can complete sentences naturally
                min_length=input_length + 10,  # At least generate a few tokens
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
                                answer_text if answer_text else ""
                            ),  # Store full answer text for fallback
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

            # Use full document text (from ChromaDB) as primary source
            # answer_text in metadata is just for fallback and should now be full length
            full_text = text if text else meta.get("answer_text", "")
            
            docs.append(
                {
                    "text": full_text,  # Always use full text, never truncated
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

        # 5) Sort by similarity and return top-K (disregarding threshold for reranking)
        # Intent-based reranking will happen in generate_response()
        docs = sorted(docs, key=lambda d: d["similarity"], reverse=True)
        
        # Return top-K chunks (default 3) for intent-based reranking later
        final_docs = docs[:top_k]

        if debug:
            print(f"DEBUG: Retrieved top-{len(final_docs)} chunks (before intent reranking):")
            for i, d in enumerate(final_docs):
                print(f"  {i+1}. sim={d['similarity']:.3f} | {d['question'][:80]}...")

        return final_docs

    def _detect_question_intent(self, query: str) -> Dict[str, any]:
        """
        Detect the intent behind the user's question to help prioritize relevant chunks.
        Returns a dict with intent type and keywords.
        """
        query_lower = query.lower().strip()
        
        intent = {
            "type": "general",  # general, definition, how_to, feature, comparison
            "main_topic": None,
            "exclude_keywords": [],
            "require_keywords": [],
        }
        
        # Detect "what is" questions - need definition chunks
        if re.match(r"^(what|tell me about|explain|describe).*", query_lower):
            intent["type"] = "definition"
            # Extract the main topic (e.g., "what is tng ewallet" -> "tng ewallet")
            match = re.search(r"(?:what is|tell me about|explain|describe)\s+(.+)", query_lower)
            if match:
                intent["main_topic"] = match.group(1).strip()
                # For "what is X", exclude chunks about related but different topics
                # e.g., if asking about "ewallet", exclude "eshop"
                if "ewallet" in intent["main_topic"] and "eshop" not in intent["main_topic"]:
                    intent["exclude_keywords"] = ["eshop", "shop", "merchandise"]
        
        # Detect "how to" questions
        elif re.match(r"^(how|how do|how can|how to).*", query_lower):
            intent["type"] = "how_to"
        
        # Detect feature questions
        elif any(word in query_lower for word in ["feature", "can i", "does it", "support"]):
            intent["type"] = "feature"
        
        return intent

    def _score_chunk_by_intent(self, doc: Dict, intent: Dict, query: str) -> float:
        """
        Score a chunk based on how well it matches the question intent.
        Returns a boost score (0.0 to 1.0) to add to similarity.
        """
        score_boost = 0.0
        question = doc.get("question", "").lower()
        text = doc.get("text", "").lower()
        query_lower = query.lower()
        
        # For definition questions, prioritize chunks that:
        # 1. Have the exact question pattern in their question field
        if intent["type"] == "definition" and intent["main_topic"]:
            main_topic = intent["main_topic"].lower()
            
            # Big boost if the chunk's question matches the user's question pattern
            if main_topic in question:
                # Check if it's a "what is X" question matching "what is X" answer
                if re.search(r"what (is|are)", question) and re.search(r"what (is|are)", query_lower):
                    score_boost += 0.5  # Strong boost for matching question pattern
                
                # Boost if the chunk question contains the main topic
                if main_topic in question:
                    score_boost += 0.3
            
            # Penalize chunks about excluded topics
            for exclude_kw in intent.get("exclude_keywords", []):
                if exclude_kw in question or exclude_kw in text:
                    score_boost -= 0.4  # Penalty for off-topic chunks
        
        # Boost chunks where the question field closely matches the user's question
        query_words = set(query_lower.split())
        question_words = set(question.split())
        # Calculate word overlap
        overlap = len(query_words.intersection(question_words))
        if len(query_words) > 0:
            overlap_ratio = overlap / len(query_words)
            score_boost += overlap_ratio * 0.2
        
        # Boost chunks that contain definition-like language
        if intent["type"] == "definition":
            definition_indicators = ["is an", "is a", "provides", "offers", "service", "application"]
            if any(indicator in text[:200] for indicator in definition_indicators):
                score_boost += 0.2
        
        return min(max(score_boost, -0.5), 0.8)  # Clamp between -0.5 and 0.8

    def _rerank_by_intent(self, docs: List[Dict], query: str) -> List[Dict]:
        """
        Re-rank documents by intent relevance, not just similarity.
        This helps prioritize chunks that actually answer the question.
        """
        intent = self._detect_question_intent(query)
        
        # Score each document
        for doc in docs:
            base_similarity = doc.get("similarity", 0.0)
            intent_boost = self._score_chunk_by_intent(doc, intent, query)
            # Create a combined score: similarity + intent boost
            doc["intent_score"] = base_similarity + intent_boost
            doc["intent_boost"] = intent_boost  # Store for debugging
        
        # Sort by intent_score (similarity + intent boost)
        reranked = sorted(docs, key=lambda d: d.get("intent_score", d.get("similarity", 0.0)), reverse=True)
        
        return reranked

    def _llm_rerank_chunks(self, docs: List[Dict], query: str) -> Dict:
        """
        Use LLM to score and select the single best chunk that answers the question.
        Returns only the top 1 chunk to prevent hallucination from multiple chunks.
        """
        if not docs:
            return None
        
        if len(docs) == 1:
            return docs[0]
        
        # Use all provided chunks (should be up to 3)
        top_chunks = docs[:min(3, len(docs))]
        
        # Build a simple ranking prompt
        chunk_descriptions = []
        for i, doc in enumerate(top_chunks):
            question = doc.get("question", "").strip()
            text_preview = doc.get("text", "").strip()[:300]  # Increased to 300 chars for better context
            chunk_descriptions.append(
                f"Chunk {i+1}:\n"
                f"Question: {question}\n"
                f"Content preview: {text_preview}..."
            )
        
        # Dynamic chunk numbers based on actual count
        chunk_numbers = ", ".join([str(i+1) for i in range(len(top_chunks))])
        
        ranking_prompt = f"""You are helping to select the single best chunk that answers a user's question.

User's Question: {query}

Available Chunks:
{chr(10).join(chunk_descriptions)}

Which chunk number ({chunk_numbers}) most directly and completely answers the user's question?
Respond with ONLY the number ({chunk_numbers}). Do not include any other text.
"""
        
        try:
            response = self._call_llm(
                ranking_prompt,
                max_new_tokens=10,  # Just need a number
                temperature=0.0,  # Deterministic
                do_sample=False,
            )
            
            # Extract the chunk number - try multiple parsing strategies
            chunk_num = None
            
            # Strategy 1: Look for standalone digit
            response_clean = response.strip()
            if response_clean.isdigit():
                chunk_num = int(response_clean)
            else:
                # Strategy 2: Extract first digit from response
                for char in response_clean:
                    if char.isdigit():
                        chunk_num = int(char)
                        break
                # Strategy 3: Look for "chunk 1" or "1" in text
                if chunk_num is None:
                    match = re.search(r'(?:chunk\s*)?(\d+)', response_clean, re.IGNORECASE)
                    if match:
                        chunk_num = int(match.group(1))
            
            if chunk_num and 1 <= chunk_num <= len(top_chunks):
                selected_chunk = top_chunks[chunk_num - 1]
                print(f"[DEBUG] LLM reranker selected chunk {chunk_num} from {len(top_chunks)} candidates")
                return selected_chunk
            else:
                # Fallback to first chunk if parsing fails
                print(f"[DEBUG] LLM reranker failed to parse response '{response_clean}', falling back to chunk 1")
                return top_chunks[0]
                
        except Exception as e:
            # On error, return the first chunk (already reranked by intent)
            print(f"[DEBUG] LLM reranker error: {e}, falling back to chunk 1")
            return top_chunks[0]

    def _filter_irrelevant_chunks(self, docs: List[Dict], query: str) -> List[Dict]:
        """
        Filter out chunks with boilerplate, irrelevant content, or very low similarity.
        """
        filtered = []
        query_lower = query.lower()
        intent = self._detect_question_intent(query)
        
        # Boilerplate patterns to exclude
        boilerplate_patterns = [
            r"Below are related articles.*",
            r"For more information.*Product Disclosure Sheet.*",
            r"Below are related articles that might be useful.*",
        ]
        
        for doc in docs:
            text = doc.get("text", "").strip()
            question = doc.get("question", "").strip()
            
            if not text or len(text) < 10:
                continue
            
            # Clean boilerplate from text instead of filtering the whole chunk
            # This preserves useful content that comes before boilerplate
            original_text = text
            for pattern in boilerplate_patterns:
                text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
            
            # Only skip if the text is now too short after cleaning (mostly boilerplate)
            text_cleaned = text.strip()
            if len(text_cleaned) < 20:
                # Debug: show which chunk was filtered due to boilerplate
                print(f"[DEBUG] Filtered chunk '{question[:60]}...' - mostly boilerplate")
                continue
            
            # Update the doc with cleaned text
            doc["text"] = text_cleaned
            
            # Filter out chunks about excluded topics (e.g., eShop when asking about eWallet)
            if intent.get("exclude_keywords"):
                is_excluded = False
                for exclude_kw in intent["exclude_keywords"]:
                    # Only exclude if the excluded keyword is prominent in the question
                    # Don't exclude if it's just mentioned in passing
                    if exclude_kw in question.lower() and intent["main_topic"] not in question.lower():
                        is_excluded = True
                        break
                if is_excluded:
                    print(f"[DEBUG] Filtered chunk '{question[:60]}...' - excluded topic")
                    continue
            
            # Keep chunks with reasonable similarity or that match query keywords
            similarity = doc.get("similarity", 0.0)
            if similarity >= 0.3:  # Only keep reasonably relevant chunks
                filtered.append(doc)
            elif any(word in text_cleaned.lower() for word in query_lower.split() if len(word) > 3):
                # Keep if it contains query keywords even with lower similarity
                filtered.append(doc)
            else:
                # Debug: show which chunk was filtered due to low similarity
                print(f"[DEBUG] Filtered chunk '{question[:60]}...' - similarity {similarity:.3f} too low")
        
        return filtered

    def generate_response(self, query: str, context_docs: List[Dict]) -> str:
        """
        Generate a synthesized, paraphrased response using multiple retrieved chunks.
        STRICTLY uses only information from retrieved chunks - no hallucination allowed.
        """

        if not context_docs:
            return (
                "I don't have information about that in my knowledge base. "
                "Please try rephrasing your question or contact TNG Digital support for assistance."
            )

        # Normalize and sort by similarity
        for doc in context_docs:
            doc["text"] = doc.get("text") or doc.get("answer") or ""
            doc["similarity"] = float(doc.get("similarity", 0.0))

        # Filter out irrelevant/boilerplate chunks
        filtered_docs = self._filter_irrelevant_chunks(context_docs, query)
        
        # Debug: show filtering results
        if len(filtered_docs) < len(context_docs):
            print(f"[DEBUG] Filtered {len(context_docs)} chunks down to {len(filtered_docs)} after removing boilerplate/irrelevant content")
        
        if not filtered_docs:
            # If all chunks were filtered, use the top one anyway
            filtered_docs = sorted(context_docs, key=lambda d: d["similarity"], reverse=True)[:1]

        # Re-rank by intent (not just similarity) - this prioritizes chunks that actually answer the question
        reranked_docs = self._rerank_by_intent(filtered_docs, query)
        
        best_chunk = self._llm_rerank_chunks(reranked_docs, query)
        
        if best_chunk:
            # Use only the single best chunk - no choice for the model to make
            selected_docs = [best_chunk]
        else:
            # Fallback to top chunk if reranking fails
            selected_docs = reranked_docs[:1]

        # Clean text in the selected doc - remove boilerplate
        best_chunk = selected_docs[0]
        best_text = best_chunk.get("text", "")
        best_text = re.sub(
            r"Below are related articles.*", "", best_text, flags=re.IGNORECASE | re.DOTALL
        )
        best_text = re.sub(
            r"For more information.*Product Disclosure Sheet.*",
            "",
            best_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        best_text = re.sub(r"\s+", " ", best_text).strip()
        best_chunk["text"] = best_text
        # Remember last best chunk for external callers (e.g. ask_tngd_bot)
        self._last_best_chunk = best_chunk

        # Small debug print for transparency about which FAQ chunk was used
        try:
            debug_question = (best_chunk.get("question") or "").strip()
            debug_url = (best_chunk.get("url") or "").strip()
            print(
                f"[DEBUG] Using best FAQ chunk -> question: '{debug_question[:120]}', url: '{debug_url}'"
            )
        except Exception:
            # Debug logging should never break main flow
            pass

# Persona prompt: single-chunk, strictly context-bound, formal tone
        persona_prompt = f"""
        You are a professional customer support representative for TNG Digital. Provide an accurate, factual response based ONLY on the information in the context below.

CRITICAL RULES (FOLLOW EXACTLY):
1. Use ONLY information explicitly stated in the context. Do NOT add, invent, assume, or speculate.
2. Do NOT paraphrase, rewrite, or reword the context. Extract and present the key information directly.
3. Use the EXACT wording from the context whenever possible. Do NOT create formal names, technical terms, or alternative descriptions.
4. CRITICAL: Keep ALL product names, brand names, and company names EXACTLY as written in the context. Do NOT:
   - Split names (e.g., "TNG" must stay as "TNG", not "T Ng")
   - Remove punctuation (e.g., "Sdn. Bhd." must stay as "Sdn. Bhd.", not "Sdn Bhd")
   - Add spaces in acronyms (e.g., "TNGD" must stay as "TNGD", not "TNG D")
   - Change capitalization (e.g., "TNG eWallet" must stay as "TNG eWallet")
   Examples: "TNG eWallet", "TNG Digital Sdn. Bhd. (TNGD)", "Touch 'n Go Card" - copy these EXACTLY.

   ===== ABSOLUTE EXAMPLES TO FOLLOW =====
   Correct:
   User: "What is TNG Digital?"
   Assistant: "TNG Digital is ..."

   Incorrect:
   User: "What is TNG Digital?"
   Assistant: "T Nagara Digital is ..."

   YOU MUST ALWAYS FOLLOW THE CORRECT PATTERN.
   Never generate alternative expansions, reinterpretations, or invented full forms for ANY name.

5. Present the information clearly and comprehensively. If the context lists multiple points, benefits, or features, include ALL of them. Use 2–4 sentences if needed to cover all key information clearly.
6. Maintain a formal, polite, and professional tone. Use clear, grammatically correct English with proper punctuation.
7. Do NOT use emojis, slang, hashtags, or overly casual language.
8. If the context does not fully answer the question, say in one sentence:
   "Based on the available information, [partial answer]. For additional details, please contact TNG Digital support."
9. Only use the context provided; ignore all other information.

User's Question: {query}

Context:
{best_text}

Your Answer (2–4 clear sentences covering all key points from the context, using exact information without paraphrasing):
"""


        try:
            response = self._call_llm(
                persona_prompt,
                # Increased max_new_tokens to allow complete sentences (2-4 sentences need ~200-400 tokens)
                max_new_tokens=300,
                temperature=0.0,  # Zero temperature for maximum factual accuracy, no hallucination
                do_sample=False,  # Deterministic output for consistency
            )

            # Clean up the response
            response = response.strip()
            
            # Remove any boilerplate that might have been generated
            response = re.sub(r"Below are related articles.*", "", response, flags=re.IGNORECASE | re.DOTALL)
            response = re.sub(r"For more information.*", "", response, flags=re.IGNORECASE | re.DOTALL)
            
            # Remove emojis, hashtags, and excessive casual language (hallucination indicators)
            response = re.sub(r"[#@][\w]+", "", response)  # Remove hashtags and @mentions
            response = re.sub(r"!{3,}", "!", response)  # Reduce excessive exclamation marks
            response = re.sub(r"\s+", " ", response).strip()
            
            # Validate response doesn't contain obvious hallucination markers
            # Check for made-up terms, excessive paraphrasing, or marketing language
            context_lower = best_text.lower()
            response_lower = response.lower()
            
            # Check for made-up terms that don't appear in context
            made_up_terms = [
                r"electronic\s*walal",  # Made-up term
                r"digital\s*finance\s*division",  # Made-up division
                r"electronically\s*managed\s*currency\s*storage",  # Over-paraphrasing
                r"proprietary\s*app\s*developed\s*exclusively",  # Made-up details
            ]
            
            hallucination_detected = False
            for pattern in made_up_terms:
                if re.search(pattern, response_lower, re.IGNORECASE):
                    hallucination_detected = True
                    break
            
            # Check for casual/marketing language that shouldn't be in formal responses
            if not hallucination_detected:
                hallucination_patterns = [
                    r"enjoying life.*?worry free",  # Casual rambling
                    r"happy shopping.*?!",  # Marketing language
                    r"team @\w+",  # Team mentions
                    r"#\w+",  # Hashtags
                    r"digital payment solutions for all",  # Marketing slogans
                ]
                for pattern in hallucination_patterns:
                    if re.search(pattern, response_lower, re.IGNORECASE):
                        hallucination_detected = True
                        break
            
            # If hallucination detected, use the original chunk text with minimal cleaning
            if hallucination_detected and selected_docs:
                response = selected_docs[0]["text"].strip()
                # Clean the fallback response
                response = re.sub(r"Below are related articles.*", "", response, flags=re.IGNORECASE | re.DOTALL)
                response = re.sub(r"For more information.*", "", response, flags=re.IGNORECASE | re.DOTALL)
                response = re.sub(r"\s+", " ", response).strip()
                # Truncate to first 4 sentences if too long (to allow comprehensive coverage)
                sentences = re.split(r"(?<=[.!?])\s+", response)
                if len(sentences) > 4:
                    response = " ".join(sentences[:4]).strip()

            # Fix incomplete sentences/words at the end and enforce short, focused answers
            if response:
                # Basic sentence splitting
                sentence_parts = re.split(r"(?<=[.!?])\s+", response)
                sentences = [s.strip() for s in sentence_parts if s.strip()]

                # Drop clearly incomplete trailing clauses (e.g. ending with 'while.')
                cleaned_sentences: List[str] = []
                for s in sentences:
                    if re.search(r"\bwhile\.$", s.strip(), re.IGNORECASE):
                        continue
                    cleaned_sentences.append(s)
                sentences = cleaned_sentences

                if sentences:
                    # Keep at most the first 4 sentences to allow comprehensive coverage of all benefits/points
                    sentences = sentences[:4]
                    response = " ".join(sentences).strip()
                else:
                    response = response.strip()

                # Normalise GOrewards naming if mentioned
                response = re.sub(r"(?i)goredeards", "GOrewards", response)
                response = re.sub(r"(?i)gorewards", "GOrewards", response)

                # Ensure response ends with proper punctuation
                if response and response[-1] not in ".!?":
                    if not response.endswith((":", ",", ";", "-")):
                        response += "."

            # Fallback if output is too short or empty
            if not response or len(response) < 15:
                # Use the most relevant chunk directly (full text, not truncated)
                best_chunk = selected_docs[0]["text"] if selected_docs else ""
                response = best_chunk.strip()
                if response and response[-1] not in ".!?":
                    response += "."

        except Exception as e:
            # On error, use the most relevant chunk directly
            if selected_docs:
                response = selected_docs[0]["text"].strip()
                if response and response[-1] not in ".!?":
                    response += "."
            else:
                response = "I apologize, but I'm unable to generate a response at this time."

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

    # Format retrieved chunks (top 3 from retrieve())
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

    # Log only top 3 retrieved chunks before synthesis
    try:
        # Only show top 3 chunks in debug
        top_3_chunks = retrieved_chunks[:3]
        debug_snapshot = {
            "question": question,
            "retrieved_chunks": top_3_chunks,
            "final_answer": "",
            "blocked": False,
        }
        print("\n[DEBUG] RAG state before synthesis (top 3 retrieved chunks):")
        print(json.dumps(debug_snapshot, ensure_ascii=False, indent=2))
    except Exception:
        # Logging should never break the main flow
        pass

    # Generate response using persona logic only (retrieval is already done)
    final_answer = rag.generate_response(sanitized_query, retrieved_docs)

    # Determine the single best chunk actually used for the answer
    best_chunk = getattr(rag, "_last_best_chunk", None)

    if best_chunk:
        retrieved_chunks = [
            {
                "text": best_chunk.get("text", ""),
                "question": best_chunk.get("question", ""),
                "url": best_chunk.get("url", ""),
                "category": _extract_primary_category(best_chunk),
                "similarity": float(best_chunk.get("similarity", 0.0)),
            }
        ]
    else:
        # Fallback: use the top retrieved doc
        top_doc = retrieved_docs[0]
        retrieved_chunks = [
            {
                "text": top_doc["text"],
                "question": top_doc["question"],
                "url": top_doc["url"],
                "category": _extract_primary_category(top_doc),
                "similarity": top_doc["similarity"],
            }
        ]
    # Validate response doesn't contain injection
    if rag.guardrails.detect_injection(final_answer):
        # Fallback to the best chunk text
        final_answer = retrieved_chunks[0]["text"]

    # Parse structured output into sections for easier consumption
    parsed = _parse_structured_answer(final_answer)

    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "final_answer": final_answer,
        "final_answer_parsed": parsed,
        "blocked": False,
    }
