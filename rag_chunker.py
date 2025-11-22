import re
from typing import List


class DocumentChunker:
    """Handles document chunking with overlap."""

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

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())

            # If adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap
                # Keep last N sentences for overlap
                overlap_sentences: List[str] = []
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
