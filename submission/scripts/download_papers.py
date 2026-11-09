"""
Script to download the exact 3 arXiv papers required by the assignment brief:
1. "Attention Is All You Need", arXiv 1706.03762v7
2. "Moshi: a speech-text foundation model for real-time dialogue", arXiv 2410.00037v1
3. "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling", arXiv 2507.07955v2
"""

import os
import urllib.request
import time
from pathlib import Path

PAPERS = [
    {
        "id": "1706.03762v7",
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762v7.pdf",
        "filename": "1706.03762v7_Attention_Is_All_You_Need.pdf"
    },
    {
        "id": "2410.00037v1",
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "url": "https://arxiv.org/pdf/2410.00037v1.pdf",
        "filename": "2410.00037v1_Moshi.pdf"
    },
    {
        "id": "2507.07955v2",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "url": "https://arxiv.org/pdf/2507.07955v2.pdf",
        "filename": "2507.07955v2_Dynamic_Chunking.pdf"
    }
]

def download_papers(target_dir: str = "data/papers"):
    dest = Path(target_dir)
    dest.mkdir(parents=True, exist_ok=True)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TheRelayResearchProject/1.0 (academic research)"
    }
    
    for paper in PAPERS:
        file_path = dest / paper["filename"]
        if file_path.exists() and file_path.stat().st_size > 10000:
            print(f"[ALREADY EXISTS] {paper['title']} ({file_path.stat().st_size / 1024:.1f} KB) at {file_path}")
            continue
            
        print(f"[DOWNLOADING] {paper['title']} ({paper['id']}) from {paper['url']}...")
        req = urllib.request.Request(paper["url"], headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp, open(file_path, "wb") as f:
                data = resp.read()
                f.write(data)
            print(f"[SUCCESS] Saved to {file_path} ({len(data) / 1024:.1f} KB)")
        except Exception as e:
            print(f"[ERROR] Failed to download {paper['title']}: {e}")
        time.sleep(1) # respectful rate limiting

if __name__ == "__main__":
    download_papers()
