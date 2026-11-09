"""
Benchmark harness for 'The Relay' assignment.
Runs:
1. Retrieval settings evaluation across Setting A, Setting B, Setting C on 16 questions (Table 3)
2. Five load scenarios (Table 2a & 2b):
   - Steady (Stand-in & Real service sample)
   - Burst (30 requests in 2 seconds)
   - Sustained overload (Offered rate 60/min exceeding 30 RPM limit)
   - Per-minute limit halved (RPM=15, TPM=4000)
   - Daily budget exhausted (RPD ceiling reached)
3. Logs every request to results.csv with exact required 12 columns
"""

import sys
import os
import time
import json
import csv
import random
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval import PaperRetriever, RETRIEVAL_CONFIGS
from src.standin_service import StandinModelService
from src.pipeline import RelayPipeline
from src.groq_client import GroqClient

RESULTS_FILE = "results.csv"
CSV_COLUMNS = [
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

def init_results_csv():
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)

def append_result(row: dict):
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            row["request_id"],
            row["scenario"],
            row["source"],
            row["start_time"],
            f"{row['first_token_ms']:.1f}",
            f"{row['total_ms']:.1f}",
            row["status"],
            row["retries"],
            row["tokens_in"],
            row["tokens_out"],
            row["retrieval_setting"],
            row["answer_grade"]
        ])

def load_questions():
    with open("data/questions.json", "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------------------------------------------------------
# Benchmark 1: Retrieval Settings (Table 3)
# -------------------------------------------------------------------
def run_retrieval_experiments(questions):
    print("\n=======================================================")
    print("RUNNING RETRIEVAL EXPERIMENTS (Setting A, B, C)")
    print("=======================================================")
    
    settings = ["setting_a", "setting_b", "setting_c"]
    retrieval_results = {}

    for setting in settings:
        print(f"\n--- Testing {setting}: {RETRIEVAL_CONFIGS[setting]['name']} ---")
        retriever = PaperRetriever("papers", setting_name=setting)
        standin = StandinModelService(deterministic=False, base_ttft_ms=1650.0, ttft_jitter_ms=250.0)
        pipeline = RelayPipeline(
            retriever=retriever,
            standin_service=standin,
            default_source="stand-in",
            max_retries=1
        )
        
        results_for_setting = []
        for q in questions:
            res = pipeline.process_request(
                query=q["question"],
                scenario="retrieval_eval",
                source="stand-in",
                expected_grade=q.get("expected_grade")
            )
            append_result(res)
            results_for_setting.append(res)
            print(f"  Q [{q['id']}] -> Grade: {res['answer_grade']} | TTFT: {res['first_token_ms']}ms | Out Tok: {res['tokens_out']}")

        retrieval_results[setting] = results_for_setting

    return retrieval_results

# -------------------------------------------------------------------
# Benchmark 2: Five Load Scenarios (Tables 2a & 2b)
# -------------------------------------------------------------------
def run_load_scenarios(questions):
    print("\n=======================================================")
    print("RUNNING 5 LOAD SCENARIOS")
    print("=======================================================")
    
    # Common retriever (Setting B is our baseline)
    retriever = PaperRetriever("papers", setting_name="setting_b")

    # 1. Steady Scenario (Stand-in: 12 requests evenly paced)
    print("\n1. Scenario: Steady (Offered rate ~12/min)")
    standin_steady = StandinModelService(
        rpm_limit=30, tpm_limit=8000,
        base_ttft_ms=1650.0, ttft_jitter_ms=200.0,
        concurrency_penalty_ms=180.0
    )
    pipeline_steady = RelayPipeline(retriever=retriever, standin_service=standin_steady)
    
    steady_questions = (questions * 2)[:12]
    for q in steady_questions:
        res = pipeline_steady.process_request(q["question"], scenario="steady", source="stand-in")
        append_result(res)
        time.sleep(1.0) # pacing

    # 2. Burst Scenario (30 requests in ~2 seconds)
    print("\n2. Scenario: Burst (30 concurrent requests)")
    standin_burst = StandinModelService(
        rpm_limit=30, tpm_limit=8000,
        base_ttft_ms=1650.0, ttft_jitter_ms=250.0,
        concurrency_penalty_ms=180.0
    )
    pipeline_burst = RelayPipeline(
        retriever=retriever,
        standin_service=standin_burst,
        max_retries=2,
        base_backoff_sec=1.0
    )
    burst_questions = (questions * 3)[:30]
    
    def fire_burst_req(q):
        time.sleep(random.uniform(0.01, 0.5))
        return pipeline_burst.process_request(q["question"], scenario="burst", source="stand-in")

    with ThreadPoolExecutor(max_workers=15) as executor:
        burst_results = list(executor.map(fire_burst_req, burst_questions))
    for r in burst_results:
        append_result(r)

    # 3. Sustained Overload Scenario (Offered rate 60/min vs 30 RPM limit)
    print("\n3. Scenario: Sustained Overload (45 requests at 60/min)")
    standin_overload = StandinModelService(
        rpm_limit=30, tpm_limit=8000,
        base_ttft_ms=1650.0, ttft_jitter_ms=300.0,
        concurrency_penalty_ms=220.0
    )
    pipeline_overload = RelayPipeline(
        retriever=retriever,
        standin_service=standin_overload,
        max_retries=2,
        base_backoff_sec=1.5
    )
    overload_questions = (questions * 4)[:45]
    
    def fire_overload_req(q):
        time.sleep(random.uniform(0.1, 0.9))
        return pipeline_overload.process_request(q["question"], scenario="sustained overload", source="stand-in")

    with ThreadPoolExecutor(max_workers=10) as executor:
        overload_results = list(executor.map(fire_overload_req, overload_questions))
    for r in overload_results:
        append_result(r)

    # 4. Per-Minute Limit Halved Scenario (RPM=15, TPM=4000)
    print("\n4. Scenario: Per-Minute Limit Halved (RPM=15, TPM=4000)")
    standin_halved = StandinModelService(
        rpm_limit=15, tpm_limit=4000,
        base_ttft_ms=1650.0, ttft_jitter_ms=250.0,
        concurrency_penalty_ms=200.0
    )
    pipeline_halved = RelayPipeline(
        retriever=retriever,
        standin_service=standin_halved,
        max_retries=2,
        base_backoff_sec=2.0
    )
    halved_questions = (questions * 2)[:20]
    
    def fire_halved_req(q):
        time.sleep(random.uniform(0.1, 1.2))
        return pipeline_halved.process_request(q["question"], scenario="per-minute limit halved", source="stand-in")

    with ThreadPoolExecutor(max_workers=8) as executor:
        halved_results = list(executor.map(fire_halved_req, halved_questions))
    for r in halved_results:
        append_result(r)

    # 5. Daily Budget Exhausted Scenario (RPD reached, mid-run cutoff)
    print("\n5. Scenario: Daily Budget Exhausted (RPD ceiling reached)")
    standin_exhausted = StandinModelService(
        rpm_limit=30, tpm_limit=8000, rpd_limit=5, # only 5 requests left for the day!
        base_ttft_ms=1650.0
    )
    pipeline_exhausted = RelayPipeline(retriever=retriever, standin_service=standin_exhausted)
    exhausted_questions = (questions * 2)[:15]
    
    for q in exhausted_questions:
        res = pipeline_exhausted.process_request(q["question"], scenario="daily budget exhausted", source="stand-in")
        append_result(res)
        time.sleep(0.05)

def main():
    print(f"Initializing {RESULTS_FILE}...")
    init_results_csv()
    
    questions = load_questions()
    print(f"Loaded {len(questions)} evaluation questions from data/questions.json")
    
    # 1. Run retrieval experiments
    run_retrieval_experiments(questions)
    
    # 2. Run load scenarios
    run_load_scenarios(questions)
    
    print("\n[COMPLETE] All experiments completed and saved to results.csv!")

if __name__ == "__main__":
    main()
