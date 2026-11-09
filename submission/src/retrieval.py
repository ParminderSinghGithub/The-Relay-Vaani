"""
Retrieval engine for the 3 research papers.
Supports:
- Three configurable settings (Setting A, Setting B, Setting C)
- TF-IDF + BM25-style lexical scoring
- Section & Title aware boosting
- Re-ranking step for Setting B
"""

import re
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.paper_parser import load_all_papers, PAPER_METADATA

# Preset settings matching Table 3 of the assignment brief
RETRIEVAL_CONFIGS = {
    "setting_a": {
        "name": "Setting A (200t, k=6, no re-rank)",
        "chunk_size_words": 150,  # ~200 tokens
        "overlap_words": 30,
        "top_k": 6,
        "re_rank": False
    },
    "setting_b": {
        "name": "Setting B (500t, k=4, re-rank yes)",
        "chunk_size_words": 380,  # ~500 tokens
        "overlap_words": 60,
        "top_k": 4,
        "candidate_k": 10,
        "re_rank": True
    },
    "setting_c": {
        "name": "Setting C (1000t, k=2, no re-rank)",
        "chunk_size_words": 750,  # ~1000 tokens
        "overlap_words": 100,
        "top_k": 2,
        "re_rank": False
    }
}

class PaperRetriever:
    def __init__(self, papers_dir: str = "papers", setting_name: str = "setting_b"):
        self.papers_dir = papers_dir
        self.set_setting(setting_name)

    def set_setting(self, setting_name: str):
        if setting_name not in RETRIEVAL_CONFIGS:
            raise ValueError(f"Unknown setting: {setting_name}. Must be one of {list(RETRIEVAL_CONFIGS.keys())}")
        self.setting_name = setting_name
        self.config = RETRIEVAL_CONFIGS[setting_name]
        
        # Load and chunk
        self.chunks = load_all_papers(
            papers_dir=self.papers_dir,
            chunk_size_words=self.config["chunk_size_words"],
            overlap_words=self.config["overlap_words"]
        )
        
        # Build TF-IDF index
        self.corpus = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_df=0.85,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def _score_re_ranking(self, query: str, candidate_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Re-ranking function that evaluates:
        1. Exact phrase and keyword density in text
        2. Keyword match in Section / Title
        3. Penalty for generic reference lists
        """
        query_terms = set(re.findall(r"\w+", query.lower()))
        query_lower = query.lower()
        
        scored_candidates = []
        for c in candidate_chunks:
            text_lower = c["text"].lower()
            section_lower = c["section"].lower()
            title_lower = c["paper_title"].lower()
            
            # Base score from first stage TF-IDF
            score = c["score"] * 1.0
            
            # Boost if query term in section header
            for term in query_terms:
                if len(term) > 3 and term in section_lower:
                    score += 0.15
                if len(term) > 3 and term in title_lower:
                    score += 0.10
                    
            # Boost for exact key phrase matches (e.g. "multi-head attention", "dynamic chunking")
            for length in [3, 2]:
                words = query_lower.split()
                for i in range(len(words) - length + 1):
                    ngram = " ".join(words[i:i+length])
                    if len(ngram) > 6 and ngram in text_lower:
                        score += 0.25
                        
            # Penalty for References / bibliography sections
            if "reference" in section_lower or "bibliography" in section_lower:
                score *= 0.3
                
            c_copy = dict(c)
            c_copy["rerank_score"] = score
            scored_candidates.append(c_copy)
            
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Tool call: retrieves top-k relevant passages from the 3 papers.
        """
        if not self.chunks:
            return []
            
        k = top_k or self.config["top_k"]
        candidate_k = self.config.get("candidate_k", k * 2) if self.config.get("re_rank") else k
        
        query_vec = self.vectorizer.transform([query])
        scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()
        
        # Get top indices
        top_indices = np.argsort(scores)[::-1][:candidate_k]
        
        candidates = []
        for idx in top_indices:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            candidates.append(chunk)
            
        if self.config.get("re_rank"):
            ranked = self._score_re_ranking(query, candidates)
            return ranked[:k]
            
        return candidates[:k]

    def format_context_for_prompt(self, passages: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved passages with citation anchors for LLM prompt.
        """
        if not passages:
            return "No passages retrieved."
            
        context_parts = []
        for i, p in enumerate(passages):
            part = (
                f"--- Passage [{i+1}] ---\n"
                f"Citation: {p['citation_tag']}\n"
                f"Paper: {p['paper_title']} ({p['paper_id']})\n"
                f"Section: {p['section']} | Page: {p['page']}\n"
                f"Content: {p['text']}\n"
            )
            context_parts.append(part)
        return "\n".join(context_parts)
