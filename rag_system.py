"""
Production-Grade RAG System with Hybrid Search
- Semantic Search: 768-dim embeddings (all-mpnet-base-v2)
- Keyword Search: BM25 for exact matches
- Re-ranking: Cross-encoder for precision
- Memory-aware context generation
"""
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from typing import List, Dict, Optional
import pickle
import os


class RAGSystem:
    """
    Hybrid RAG System combining semantic and keyword search

    Features:
    - 768 dimensional embeddings for semantic understanding
    - BM25 for exact keyword matching
    - Cross-encoder re-ranking for accuracy
    - Overlapping chunks for context continuity

    Usage:
        rag = RAGSystem()
        rag.build_index("video_123", transcript)
        results = rag.search("What is recursion?", top_k=5, strategy="hybrid")
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize hybrid RAG system

        Args:
            model_name: Sentence transformer model (default: all-mpnet-base-v2)
        """
        print(f" Loading RAG models...")

        self.embedding_model = SentenceTransformer(model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()

        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        print(f" RAG models loaded!")
        print(f"   Embeddings: {self.dimension}-dimensional")
        print(f"   Re-ranking: Enabled")
        print(f"   Hybrid Search: Ready")

        self.faiss_index = None
        self.bm25_index = None
        self.chunks = []
        self.video_id = None
        self.video_title = ""

    def create_chunks(
        self,
        transcript: List[Dict],
        chunk_size: int = 5,
        overlap: int = 2
    ) -> List[Dict]:
        """Create overlapping chunks from transcript
        Args:
            transcript: List of transcript segments
            chunk_size: Number of segments per chunk (default: 5)
            overlap: Number of overlapping segments (default: 2)
            
        Returns:
            List of chunks with metadata"""
        chunks = []
        step = chunk_size - overlap

        for i in range(0, len(transcript), step):
            segments = transcript[i:i + chunk_size]
            if not segments:
                continue

            text = " ".join([s['text'] for s in segments])

            chunks.append({
                "text": text.strip(),
                "start_time": segments[0]['start'],
                "end_time": segments[-1]['start'] + segments[-1]['duration'],
                "segments": segments,
                "chunk_index": len(chunks),
                "segment_count": len(segments)
            })

        return chunks

    def build_index(
        self,
        video_id: str,
        transcript: List[Dict],
        video_title: str = ""
    ):
        """Build hybrid index (FAISS + BM25)
        Args:
            video_id: YouTube video ID
            transcript: List of transcript segments
            video_title: Optional video title
            """
        print(f"\n{'='*70}")
        print(f"🔨 Building Hybrid RAG Index")
        print(f"{'='*70}")
        print(f"Video ID: {video_id}")
        print(f"Video Title: {video_title or 'Unknown'}")
        print(f"Total Segments: {len(transcript)}")

        self.video_id = video_id
        self.video_title = video_title

        print(f"\n1️⃣ Creating chunks...")
        self.chunks = self.create_chunks(transcript)
        print(f"   Created {len(self.chunks)} chunks")

        if not self.chunks:
            print(" No chunks created!")
            return

        texts = [chunk['text'] for chunk in self.chunks]

        print(f"\n Building FAISS index (semantic)...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )

        self.faiss_index = faiss.IndexFlatL2(self.dimension)
        self.faiss_index.add(np.array(embeddings).astype('float32'))
        print(f"   FAISS index built ({self.faiss_index.ntotal} vectors)")

        print(f"\n Building BM25 index (keywords)...")
        tokenized_chunks = [text.lower().split() for text in texts]
        self.bm25_index = BM25Okapi(tokenized_chunks)
        print(f"   BM25 index built")

        print(f"\n{'='*70}")
        print(f" Hybrid Index Complete!")
        print(f"   - Semantic vectors: {self.faiss_index.ntotal}")
        print(f"   - Keyword index: Ready")
        print(f"   - Re-ranker: Ready")
        print(f"{'='*70}")

    def search_semantic(self, query: str, top_k: int = 10) -> List[Dict]:
        """Semantic search using embeddings"""
        query_embedding = self.embedding_model.encode([query])

        distances, indices = self.faiss_index.search(
            np.array(query_embedding).astype('float32'),
            top_k
        )

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    "text": chunk['text'],
                    "start_time": chunk['start_time'],
                    "end_time": chunk['end_time'],
                    "semantic_score": float(1 / (1 + distance)),
                    "chunk_index": chunk['chunk_index']
                })

        return results

    def search_bm25(self, query: str, top_k: int = 10) -> List[Dict]:
        """Keyword search using BM25"""
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)

        top_indices = np.argsort(bm25_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    "text": chunk['text'],
                    "start_time": chunk['start_time'],
                    "end_time": chunk['end_time'],
                    "bm25_score": float(bm25_scores[idx]),
                    "chunk_index": chunk['chunk_index']
                })

        return results

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5
    ) -> List[Dict]:
        """Hybrid search combining semantic and keyword matching
        Args:
            query: Search query
            top_k: Number of candidates to fetch
            alpha: Weight for semantic (0.5 = equal weight)
                   1.0 = only semantic, 0.0 = only BM25
        
        Returns:
            List of results with combined scores
        
        """
        semantic_results = self.search_semantic(query, top_k=top_k * 2)
        bm25_results = self.search_bm25(query, top_k=top_k * 2)

        semantic_scores = np.array([r['semantic_score'] for r in semantic_results])
        semantic_scores = semantic_scores / (semantic_scores.max() + 1e-6)

        bm25_scores = np.array([r['bm25_score'] for r in bm25_results])
        bm25_scores = bm25_scores / (bm25_scores.max() + 1e-6)

        combined = {}

        for i, result in enumerate(semantic_results):
            idx = result['chunk_index']
            combined[idx] = {
                **result,
                'semantic_score_norm': float(semantic_scores[i]),
                'bm25_score_norm': 0.0
            }

        for i, result in enumerate(bm25_results):
            idx = result['chunk_index']
            if idx in combined:
                combined[idx]['bm25_score_norm'] = float(bm25_scores[i])
                combined[idx]['bm25_score'] = result['bm25_score']
            else:
                combined[idx] = {
                    **result,
                    'semantic_score_norm': 0.0,
                    'bm25_score_norm': float(bm25_scores[i])
                }

        for chunk in combined.values():
            chunk['hybrid_score'] = (
                alpha * chunk['semantic_score_norm'] +
                (1 - alpha) * chunk['bm25_score_norm']
            )

        results = sorted(
            combined.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )[:top_k]

        return results

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """Re-rank candidates using cross-encoder
        rgs:
            query: Search query
            candidates: Initial results
            top_k: Final number of results
            
        Returns:
            Re-ranked results
        """
        if not candidates:
            return []

        pairs = [[query, c['text']] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        for candidate, score in zip(candidates, rerank_scores):
            candidate['rerank_score'] = float(score)

        results = sorted(
            candidates,
            key=lambda x: x['rerank_score'],
            reverse=True
        )[:top_k]

        for i, result in enumerate(results, 1):
            result['rank'] = i

        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "hybrid_rerank",
        alpha: float = 0.5
    ) -> List[Dict]:
        """Main search method with multiple strategies
        
        rgs:
            query: Search query
            top_k: Number of results
            strategy: Search strategy
                - 'semantic': Pure embedding search
                - 'bm25': Pure keyword search
                - 'hybrid': Combine semantic + BM25
                - 'hybrid_rerank': Hybrid + re-ranking (RECOMMENDED)
            alpha: Weight for hybrid (0.5 = equal, 1.0 = only semantic)
            
        Returns:
            List of ranked results
        """
        if self.faiss_index is None or self.bm25_index is None:
            raise ValueError("Index not built. Call build_index() first.")

        if strategy == "semantic":
            results = self.search_semantic(query, top_k=top_k)
            for i, r in enumerate(results, 1):
                r['rank'] = i
            return results

        elif strategy == "bm25":
            results = self.search_bm25(query, top_k=top_k)
            for i, r in enumerate(results, 1):
                r['rank'] = i
            return results

        elif strategy == "hybrid":
            results = self.search_hybrid(query, top_k=top_k, alpha=alpha)
            for i, r in enumerate(results, 1):
                r['rank'] = i
            return results

        elif strategy == "hybrid_rerank":
            candidates = self.search_hybrid(query, top_k=top_k * 3, alpha=alpha)
            results = self.rerank(query, candidates, top_k=top_k)
            return results

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def get_context(
        self,
        query: str,
        top_k: int = 3,
        max_tokens: int = 1500,
        strategy: str = "hybrid_rerank"
    ) -> Dict:
        """Get formatted context for LLM
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            max_tokens: Maximum tokens (chars/4)
            strategy: Search strategy
            
        Returns:
            Dictionary with formatted context
            """
        results = self.search(query, top_k=top_k, strategy=strategy)

        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4

        for result in results:
            timestamp = self._format_timestamp(result['start_time'])
            formatted = f"[{timestamp}] {result['text']}"

            if total_chars + len(formatted) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    formatted = formatted[:remaining] + "..."
                else:
                    break

            context_parts.append({
                "text": formatted,
                "timestamp": result['start_time'],
                "rank": result['rank']
            })
            total_chars += len(formatted)

        context_text = "\n\n".join([p['text'] for p in context_parts])

        return {
            "context": context_text,
            "num_chunks": len(context_parts),
            "total_chars": total_chars,
            "chunks": context_parts,
            "video_id": self.video_id,
            "video_title": self.video_title,
            "strategy": strategy
        }

    def get_stats(self) -> Dict:
        """Get system statistics"""
        if self.faiss_index is None or self.bm25_index is None:
            return {
                "status": "not_built",
                "message": "Index not built yet"
            }

        avg_segments = np.mean([c['segment_count'] for c in self.chunks]) if self.chunks else 0

        return {
            "status": "ready",
            "video_id": self.video_id,
            "video_title": self.video_title,
            "total_chunks": len(self.chunks),
            "faiss_index_size": self.faiss_index.ntotal,
            "bm25_index_size": len(self.chunks),
            "dimension": self.dimension,
            "avg_segments_per_chunk": float(avg_segments),
            "features": [
                "768-dim semantic embeddings",
                "BM25 keyword search",
                "Hybrid search (50/50 default)",
                "Cross-encoder re-ranking",
                "Overlapping chunks"
            ]
        }

    def save_index(self, filepath: str):
        """Save indexes to disk"""
        if self.faiss_index is None or self.bm25_index is None:
            raise ValueError("No index to save")

        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        faiss.write_index(self.faiss_index, f"{filepath}.faiss")

        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump({
                'bm25_index': self.bm25_index,
                'chunks': self.chunks,
                'video_id': self.video_id,
                'video_title': self.video_title,
                'dimension': self.dimension
            }, f)

        print(f" Index saved to {filepath}")

    def load_index(self, filepath: str):
        """Load indexes from disk"""
        self.faiss_index = faiss.read_index(f"{filepath}.faiss")

        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
            self.bm25_index = data['bm25_index']
            self.chunks = data['chunks']
            self.video_id = data['video_id']
            self.video_title = data.get('video_title', '')

        print(f" Index loaded from {filepath}")

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Convert seconds to MM:SS or HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
