"""
Paper ingestion and chunking module for the 3 arXiv papers:
- 1706.03762v7: Attention Is All You Need
- 2410.00037v1: Moshi: a speech-text foundation model for real-time dialogue
- 2507.07955v2: Dynamic Chunking for End-to-End Hierarchical Sequence Modeling
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any
import pypdf

CACHE_DIR = Path("cache")

PAPER_METADATA = {
    "1706.03762v7.pdf": {
        "paper_id": "1706.03762v7",
        "title": "Attention Is All You Need",
        "short_title": "Attention",
        "authors": "Vaswani et al."
    },
    "2410.00037v1.pdf": {
        "paper_id": "2410.00037v1",
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "short_title": "Moshi",
        "authors": "Défossez et al."
    },
    "2507.07955v2.pdf": {
        "paper_id": "2507.07955v2",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "short_title": "Dynamic Chunking",
        "authors": "Hwang et al."
    }
}

SECTION_REGEX = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9\s,\-:]{2,50})|Abstract|References|Conclusions?|Appendix(?:\s+[A-Z])?)$"
)

def clean_text(text: str) -> str:
    # Fix hyphenated line breaks e.g. "atten-\ntion" -> "attention"
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Replace single linebreaks with space if not empty line
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned_lines.append(stripped)
    return " ".join(cleaned_lines)

def extract_paper_pages(pdf_path: str, use_cache: bool = True) -> List[Dict[str, Any]]:
    pdf_name = Path(pdf_path).name
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{pdf_name}.json"
    
    if use_cache and cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
            
    reader = pypdf.PdfReader(pdf_path)
    pages_data = []
    current_section = "Introduction"
    
    for i, page in enumerate(reader.pages):
        page_num = i + 1
        raw_text = page.extract_text() or ""
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        
        # Check lines for section header updates
        for line in lines:
            if SECTION_REGEX.match(line):
                # Update current section
                current_section = line
                
        cleaned = clean_text(raw_text)
        pages_data.append({
            "page_number": page_num,
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "section": current_section
        })
        
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, ensure_ascii=False)
        
    return pages_data

def chunk_paper(
    pdf_path: str,
    meta: Dict[str, str],
    chunk_size_words: int = 350,
    overlap_words: int = 50
) -> List[Dict[str, Any]]:
    """
    Chunks a paper with exact metadata and citation string.
    chunk_size_words approximates token counts (~1 word ≈ 1.3 tokens).
    """
    pages_data = extract_paper_pages(pdf_path)
    chunks = []
    
    for page_info in pages_data:
        page_num = page_info["page_number"]
        section = page_info["section"]
        text = page_info["cleaned_text"]
        
        words = text.split()
        if not words:
            continue
            
        step = max(chunk_size_words - overlap_words, 1)
        for start_idx in range(0, len(words), step):
            end_idx = min(start_idx + chunk_size_words, len(words))
            chunk_words = words[start_idx:end_idx]
            if len(chunk_words) < 30 and chunks:
                # Append tiny leftovers to previous chunk if from same page
                if chunks[-1]["page"] == page_num:
                    chunks[-1]["text"] += " " + " ".join(chunk_words)
                    continue
                    
            chunk_text = " ".join(chunk_words)
            chunk_id = f"{meta['paper_id']}_p{page_num}_{start_idx // step}"
            citation_tag = f"[{meta['title']}, Section: {section}, Page {page_num}]"
            
            chunks.append({
                "chunk_id": chunk_id,
                "paper_id": meta["paper_id"],
                "paper_title": meta["title"],
                "short_title": meta["short_title"],
                "page": page_num,
                "section": section,
                "text": chunk_text,
                "citation_tag": citation_tag,
                "word_count": len(chunk_words),
                "approx_tokens": int(len(chunk_words) * 1.3)
            })
            
            if end_idx == len(words):
                break
                
    return chunks

def load_all_papers(
    papers_dir: str = "papers",
    chunk_size_words: int = 350,
    overlap_words: int = 50
) -> List[Dict[str, Any]]:
    all_chunks = []
    pdir = Path(papers_dir)
    for filename, meta in PAPER_METADATA.items():
        pdf_path = pdir / filename
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing required paper: {pdf_path}")
        chunks = chunk_paper(str(pdf_path), meta, chunk_size_words, overlap_words)
        all_chunks.extend(chunks)
    return all_chunks
