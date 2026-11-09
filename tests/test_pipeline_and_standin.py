"""
Unit test suite verifying:
1. Normal cited answers
2. Out-of-scope questions (Step 1 branching)
3. Unanswerable questions (clean refusal)
4. Invalid citations & bounded refinement loop (Step 4 verifier feedback)
5. Timeout handling & status
6. Rate limit (HTTP 429) & backoff retry
7. Concurrent requests & concurrency-based slowdown
"""

import sys
import os
from concurrent.futures import ThreadPoolExecutor

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval import PaperRetriever
from src.standin_service import StandinModelService
from src.pipeline import RelayPipeline

def get_test_pipeline():
    retriever = PaperRetriever("papers", setting_name="setting_b")
    standin = StandinModelService(deterministic=True, base_ttft_ms=10.0, concurrency_penalty_ms=25.0)
    pipeline = RelayPipeline(
        retriever=retriever,
        standin_service=standin,
        default_source="stand-in",
        max_retries=2,
        base_backoff_sec=0.01,
        max_backoff_sec=0.05
    )
    return pipeline

def test_normal_cited_answer():
    pipeline = get_test_pipeline()
    query = "What is the dimension d_k in multi-head attention?"
    res = pipeline.process_request(query, scenario="steady")
    
    assert res["status"] == "ok"
    assert res["answer_grade"] == "correct"
    assert "[" in res["final_answer"] and "]" in res["final_answer"]
    assert "Attention Is All You Need" in res["final_answer"]
    assert res["first_token_ms"] > 0
    assert res["total_ms"] >= res["first_token_ms"]
    assert res["tokens_in"] > 0
    assert res["tokens_out"] > 0
    assert res["retries"] == 0

def test_out_of_scope_question():
    pipeline = get_test_pipeline()
    query = "What learning rate was used for Llama 3?"
    res = pipeline.process_request(query, scenario="steady")
    
    assert res["status"] == "ok"
    assert res["answer_grade"] == "refused_correctly"
    assert "The library covers only three papers" in res["final_answer"]
    assert res["retries"] == 0
    assert res["waterfall"]["waiting_ms"] == 0.0

def test_unanswerable_question():
    pipeline = get_test_pipeline()
    query = "What is the training cost in US dollars for the 1706.03762 Transformer?"
    res = pipeline.process_request(query, scenario="steady")
    
    assert res["status"] == "ok"
    assert res["answer_grade"] == "refused_correctly"
    assert any(term in res["final_answer"].lower() for term in ["outside this scope", "do not provide", "do not say"])

def test_invalid_citation_and_refinement_loop():
    pipeline = get_test_pipeline()
    query = "What is the dimension d_k in multi-head attention?"
    res = pipeline.process_request(
        query,
        scenario="steady",
        simulate_initial_bad_citation=True
    )
    
    assert res["status"] == "ok"
    assert res["retries"] == 1, "Expected 1 loop retry from verifier feedback"
    assert "[" in res["final_answer"] and "]" in res["final_answer"]
    assert res["waterfall"]["retrying_ms"] > 0

def test_timeout_handling():
    pipeline = get_test_pipeline()
    query = "What is the dimension d_k in multi-head attention?"
    res = pipeline.process_request(
        query,
        scenario="steady",
        force_error="timeout"
    )
    
    assert res["status"] == "timeout"
    assert res["retries"] == pipeline.max_retries

def test_rate_limit_and_backoff_retry():
    pipeline = get_test_pipeline()
    query = "What is the frame rate of the Mimi codec?"
    res = pipeline.process_request(
        query,
        scenario="steady",
        force_error="rate_limit_once"
    )
    
    assert res["status"] == "ok"
    assert res["retries"] == 1
    assert "Mimi" in res["final_answer"]
    assert res["waterfall"]["waiting_ms"] > 0

def test_concurrent_requests_slowdown():
    standin = StandinModelService(
        deterministic=False,
        base_ttft_ms=50.0,
        ttft_jitter_ms=5.0,
        concurrency_penalty_ms=100.0
    )
    
    messages = [
        {"role": "system", "content": "You are a research assistant."},
        {"role": "user", "content": "What is Mimi frame rate?"}
    ]
    
    def fire_request(i):
        return standin.stream_chat(messages, query="What is Mimi frame rate?")

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(fire_request, range(5)))
        
    assert len(results) == 5
    assert all(r["status"] == "ok" for r in results)
    assert standin.peak_queue_length >= 4, f"Expected peak queue >= 4, got {standin.peak_queue_length}"
    
    latencies = [r["total_ms"] for r in results]
    min_lat = min(latencies)
    max_lat = max(latencies)
    print(f"Concurrent latencies: min={min_lat:.1f}ms, max={max_lat:.1f}ms, peak_queue={standin.peak_queue_length}")
    assert max_lat > min_lat, "Concurrency delay should cause later requests to have higher latency"

if __name__ == "__main__":
    print("Running tests manually...")
    test_normal_cited_answer()
    print("  [PASS] test_normal_cited_answer")
    test_out_of_scope_question()
    print("  [PASS] test_out_of_scope_question")
    test_unanswerable_question()
    print("  [PASS] test_unanswerable_question")
    test_invalid_citation_and_refinement_loop()
    print("  [PASS] test_invalid_citation_and_refinement_loop")
    test_timeout_handling()
    print("  [PASS] test_timeout_handling")
    test_rate_limit_and_backoff_retry()
    print("  [PASS] test_rate_limit_and_backoff_retry")
    test_concurrent_requests_slowdown()
    print("  [PASS] test_concurrent_requests_slowdown")
    print("ALL TESTS PASSED!")
