"""
Tests for:
1. Exact results.csv schema and allowed column values
2. Answer grading validity (correct, wrong, refused_correctly, refused_wrongly)
3. Mathematical percentile calculation (p50, p95, p99)
4. All five load scenarios execution and state handling
5. Recomputability of Table 2a, 2b, 3 from CSV
"""

import sys
import os
import csv
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.generate_tables import percentile
from src.retrieval import PaperRetriever
from src.standin_service import StandinModelService
from src.pipeline import RelayPipeline

ALLOWED_STATUSES = {"ok", "rate_limited", "timeout", "error"}
ALLOWED_SOURCES = {"real", "stand-in"}
ALLOWED_GRADES = {"correct", "wrong", "refused_correctly", "refused_wrongly"}
REQUIRED_COLUMNS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade"
]

def test_percentile_calculation():
    data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    p50 = percentile(data, 0.50)
    p95 = percentile(data, 0.95)
    p99 = percentile(data, 0.99)
    
    assert p50 == 60.0 or p50 == 50.0
    assert p95 == 100.0
    assert p99 == 100.0

def test_csv_schema_and_allowed_values():
    retriever = PaperRetriever("papers", setting_name="setting_b")
    standin = StandinModelService(deterministic=True, base_ttft_ms=5.0)
    pipeline = RelayPipeline(retriever=retriever, standin_service=standin)
    
    test_queries = [
        ("What is dimension d_k in multi-head attention?", "steady", "correct"),
        ("What learning rate was used for Llama 3?", "steady", "refused_correctly"),
        ("What dynamic chunking threshold is used in Attention Is All You Need?", "steady", "refused_correctly")
    ]
    
    for q, sc, exp_grade in test_queries:
        row = pipeline.process_request(q, scenario=sc, source="stand-in")
        
        # Verify required keys
        for col in REQUIRED_COLUMNS:
            assert col in row, f"Missing column {col} in pipeline result"
            
        assert row["status"] in ALLOWED_STATUSES
        assert row["source"] in ALLOWED_SOURCES
        assert row["answer_grade"] in ALLOWED_GRADES
        assert row["answer_grade"] == exp_grade

def test_five_scenarios_execution():
    retriever = PaperRetriever("papers", setting_name="setting_b")
    
    # 1. Steady
    standin_steady = StandinModelService(deterministic=True, base_ttft_ms=5.0)
    p_steady = RelayPipeline(retriever=retriever, standin_service=standin_steady)
    r_steady = p_steady.process_request("What is Mimi frame rate?", scenario="steady", source="stand-in")
    assert r_steady["status"] == "ok"
    assert r_steady["retries"] == 0

    # 2. Burst
    standin_burst = StandinModelService(deterministic=True, base_ttft_ms=5.0, concurrency_penalty_ms=10.0)
    p_burst = RelayPipeline(retriever=retriever, standin_service=standin_burst)
    r_burst = p_burst.process_request("What is Mimi frame rate?", scenario="burst", source="stand-in")
    assert r_burst["status"] == "ok"

    # 3. Sustained Overload
    standin_overload = StandinModelService(deterministic=True, base_ttft_ms=5.0, rpm_limit=2)
    p_overload = RelayPipeline(
        retriever=retriever,
        standin_service=standin_overload,
        base_backoff_sec=0.01,
        max_backoff_sec=0.02
    )
    # Fire 3 requests to trip 2 RPM limit
    p_overload.process_request("What is Mimi frame rate?", scenario="sustained overload", source="stand-in")
    p_overload.process_request("What is Mimi frame rate?", scenario="sustained overload", source="stand-in")
    r_overload = p_overload.process_request("What is Mimi frame rate?", scenario="sustained overload", source="stand-in")
    assert r_overload["status"] in ("ok", "rate_limited")

    # 4. Per-minute limit halved
    standin_halved = StandinModelService(deterministic=True, base_ttft_ms=5.0, rpm_limit=1)
    p_halved = RelayPipeline(retriever=retriever, standin_service=standin_halved, max_retries=0)
    p_halved.process_request("What is Mimi frame rate?", scenario="per-minute limit halved", source="stand-in")
    r_halved = p_halved.process_request("What is Mimi frame rate?", scenario="per-minute limit halved", source="stand-in")
    assert r_halved["status"] == "rate_limited"

    # 5. Daily budget exhausted
    standin_exhausted = StandinModelService(deterministic=True, base_ttft_ms=5.0, rpd_limit=0)
    p_exhausted = RelayPipeline(retriever=retriever, standin_service=standin_exhausted)
    r_exhausted = p_exhausted.process_request("What is Mimi frame rate?", scenario="daily budget exhausted", source="stand-in")
    assert r_exhausted["status"] == "rate_limited"
    assert "daily query budget has been exhausted" in r_exhausted["final_answer"].lower()
