"""
Baseline probe script for Groq API.
Sends a modest number of real requests (5 requests) with pacing to:
1. Measure latency spread (TTFT and total time)
2. Extract real rate limit headers (RPM, TPM, RPD, reset windows)
3. Protect free quota from being burned
"""

import sys
import os
import time
import statistics

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.groq_client import GroqClient

TEST_PROMPTS = [
    "What are the main components of the Transformer architecture in 1706.03762 in two concise sentences?",
    "What is the frame rate of the Mimi codec in the Moshi paper in one sentence?",
    "Explain what dynamic chunking accomplishes in H-Net in one sentence.",
    "Define scaled dot-product attention in one sentence.",
    "State the primary benefit of byte-level modeling over BPE tokenization in one sentence."
]

def run_probe(model_override=None):
    # If model is not set or defaulted to llama-3.1-8b-instant which is unavailable on user account,
    # use openai/gpt-oss-20b or qwen/qwen3.8-27b from user's active limits table
    model = model_override or os.environ.get("GROQ_MODEL")
    if not model or model == "llama-3.1-8b-instant":
        model = "openai/gpt-oss-20b"
        
    client = GroqClient(model=model)
    if not client.is_configured():
        print("[WARNING] GROQ_API_KEY is not configured or placeholder. Set it in .env to run real probe.")
        return

    print(f"Starting Groq Baseline Probe using model: {client.model}")
    print(f"Sending {len(TEST_PROMPTS)} paced queries to measure real service latency and headers...\n")

    results = []
    captured_headers = {}

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"[{i+1}/{len(TEST_PROMPTS)}] Sending query: '{prompt[:45]}...'")
        messages = [
            {"role": "system", "content": "You are a concise research paper assistant. Cite exact facts."},
            {"role": "user", "content": prompt}
        ]
        
        res = client.stream_chat(messages, max_tokens=150)
        status = res["status"]
        ttft = res["first_token_ms"]
        total = res["total_ms"]
        
        print(f"   -> Status: {status} | TTFT: {ttft:.1f}ms | Total: {total:.1f}ms | Out tokens: {res['tokens_out']}")
        if res.get("headers"):
            captured_headers = res["headers"]
            
        results.append(res)
        time.sleep(2)  # Respectful 2-second pacing to avoid any RPM pressure

    successful = [r for r in results if r["status"] == "ok"]
    if successful:
        ttfts = [r["first_token_ms"] for r in successful]
        totals = [r["total_ms"] for r in successful]
        
        ttfts.sort()
        totals.sort()
        p50_ttft = statistics.median(ttfts)
        p95_ttft = ttfts[int(len(ttfts) * 0.95)] if len(ttfts) > 1 else ttfts[0]
        
        p50_tot = statistics.median(totals)
        p95_tot = totals[int(len(totals) * 0.95)] if len(totals) > 1 else totals[0]
        
        print("\n=== Real Service Measurement Summary ===")
        print(f"Samples: {len(successful)} successful requests")
        print(f"TTFT: p50 = {p50_ttft:.1f}ms, p95 = {p95_ttft:.1f}ms")
        print(f"Total Latency: p50 = {p50_tot:.1f}ms, p95 = {p95_tot:.1f}ms")
        print("\n=== Live Rate Limit Headers Captured ===")
        for k, v in captured_headers.items():
            print(f"  {k}: {v}")
    else:
        print("\n[NOTICE] No successful responses recorded. Check client status:", results[0] if results else "No results")

if __name__ == "__main__":
    run_probe()
