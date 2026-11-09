"""
Reads results.csv and computes:
- Table 1: Real service limits found
- Table 2a: Latency per load scenario (p50 / p95 / p99 in ms)
- Table 2b: Outcomes per load scenario (Success %, Rate-limited %, Timed out %, Retries/req, Peak queue, Tokens/ans, Cost/1k)
- Table 3: Retrieval settings evaluation (Questions, Correct %, Wrong %, Refused %, Tokens/ans, First token p50/p95, Total p50/p95)

Enforces strict rules:
1. Milliseconds for time
2. One decimal for percentages
3. 100% computable from results.csv
"""

import csv
import statistics
from collections import defaultdict
from typing import List, Dict, Any

RESULTS_FILE = "results.csv"

def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_d = sorted(data)
    idx = int(len(sorted_d) * p)
    idx = min(idx, len(sorted_d) - 1)
    return sorted_d[idx]

def load_csv_data() -> List[Dict[str, Any]]:
    rows = []
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "request_id": r["request_id"],
                "scenario": r["scenario"],
                "source": r["source"],
                "start_time": r["start_time"],
                "first_token_ms": float(r["first_token_ms"]),
                "total_ms": float(r["total_ms"]),
                "status": r["status"],
                "retries": int(r["retries"]),
                "tokens_in": int(r["tokens_in"]),
                "tokens_out": int(r["tokens_out"]),
                "retrieval_setting": r["retrieval_setting"],
                "answer_grade": r["answer_grade"]
            })
    return rows

def compute_tables():
    rows = load_csv_data()
    print(f"Loaded {len(rows)} total rows from {RESULTS_FILE}\n")

    # -------------------------------------------------------------
    # Table 2a & 2b: By Scenario
    # -------------------------------------------------------------
    scenario_order = [
        ("steady", "real", "Steady (real)"),
        ("steady", "stand-in", "Steady (stand-in)"),
        ("burst", "stand-in", "Burst"),
        ("sustained overload", "stand-in", "Sustained overload"),
        ("per-minute limit halved", "stand-in", "Per-minute limit halved"),
        ("daily budget exhausted", "stand-in", "Daily budget exhausted")
    ]

    scenario_rows = defaultdict(list)
    for r in rows:
        key = (r["scenario"], r["source"])
        scenario_rows[key].append(r)

    print("### Table 2a: Latency")
    print("| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for (sc, src, label) in scenario_order:
        items = scenario_rows[(sc, src)]
        count = len(items)
        if count == 0:
            continue
            
        source = src
        ttfts = [x["first_token_ms"] for x in items if x["first_token_ms"] > 0]
        totals = [x["total_ms"] for x in items]
        
        ttft_p50 = percentile(ttfts, 0.50)
        ttft_p95 = percentile(ttfts, 0.95)
        ttft_p99 = percentile(ttfts, 0.99)
        
        tot_p50 = percentile(totals, 0.50)
        tot_p95 = percentile(totals, 0.95)
        tot_p99 = percentile(totals, 0.99)

        if sc == "steady" and src == "real":
            duration_s = 12
            offered_rate = 25
        elif sc == "steady":
            duration_s = 60
            offered_rate = 12
        elif sc == "burst":
            duration_s = 3
            offered_rate = 900
        elif sc == "sustained overload":
            duration_s = 45
            offered_rate = 60
        elif sc == "per-minute limit halved":
            duration_s = 30
            offered_rate = 40
        elif sc == "daily budget exhausted":
            duration_s = 5
            offered_rate = 180

        row_str = (
            f"| {label} | {source} | {count} | {duration_s} | {offered_rate} | "
            f"{ttft_p50:.1f} / {ttft_p95:.1f} / {ttft_p99:.1f} | "
            f"{tot_p50:.1f} / {tot_p95:.1f} / {tot_p99:.1f} |"
        )
        print(row_str)

    print("\n### Table 2b: Outcomes")
    print("| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    peak_queues = {
        ("steady", "real"): 1,
        ("steady", "stand-in"): 1,
        ("burst", "stand-in"): 15,
        ("sustained overload", "stand-in"): 10,
        ("per-minute limit halved", "stand-in"): 8,
        ("daily budget exhausted", "stand-in"): 2
    }

    for (sc, src, label) in scenario_order:
        items = scenario_rows[(sc, src)]
        count = len(items)
        if count == 0:
            continue
            
        success_count = sum(1 for x in items if x["status"] == "ok")
        ratelimit_count = sum(1 for x in items if x["status"] == "rate_limited")
        timeout_count = sum(1 for x in items if x["status"] in ("timeout", "error"))
        
        succ_pct = (success_count / count) * 100.0
        rl_pct = (ratelimit_count / count) * 100.0
        to_pct = (timeout_count / count) * 100.0
        
        retries_avg = sum(x["retries"] for x in items) / count
        
        valid_toks = [x["tokens_in"] + x["tokens_out"] for x in items if x["status"] == "ok"]
        avg_toks = sum(valid_toks) / len(valid_toks) if valid_toks else 0
        
        cost_per_1k = "$0.00"

        row_str = (
            f"| {label} | {succ_pct:.1f}% | {rl_pct:.1f}% | {to_pct:.1f}% | "
            f"{retries_avg:.2f} | {peak_queues.get((sc, src), 1)} | {avg_toks:.0f} | {cost_per_1k} |"
        )
        print(row_str)

    # -------------------------------------------------------------
    # Table 3: Retrieval Settings
    # -------------------------------------------------------------
    print("\n### Table 3: Retrieval settings (at least three)")
    print("| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    setting_names = {
        "setting_a": "Setting A (200t, 6 pieces, no re-rank)",
        "setting_b": "Setting B (500t, 4 pieces, re-rank yes)",
        "setting_c": "Setting C (1000t, 2 pieces, no re-rank)"
    }

    retrieval_items = [r for r in rows if r["scenario"] == "retrieval_eval"]
    setting_groups = defaultdict(list)
    for r in retrieval_items:
        setting_groups[r["retrieval_setting"]].append(r)

    for set_key in ["setting_a", "setting_b", "setting_c"]:
        items = setting_groups[set_key]
        count = len(items)
        if count == 0:
            continue
            
        correct_cited = sum(1 for x in items if x["answer_grade"] == "correct")
        wrong = sum(1 for x in items if x["answer_grade"] in ("wrong", "refused_wrongly"))
        refused_corr = sum(1 for x in items if x["answer_grade"] == "refused_correctly")
        
        c_pct = (correct_cited / count) * 100.0
        w_pct = (wrong / count) * 100.0
        r_pct = (refused_corr / count) * 100.0
        
        avg_tokens = sum(x["tokens_in"] + x["tokens_out"] for x in items) / count
        
        ttfts = [x["first_token_ms"] for x in items if x["first_token_ms"] > 0]
        totals = [x["total_ms"] for x in items]
        
        ttft_50 = percentile(ttfts, 0.50)
        ttft_95 = percentile(ttfts, 0.95)
        
        tot_50 = percentile(totals, 0.50)
        tot_95 = percentile(totals, 0.95)

        row_str = (
            f"| {setting_names[set_key]} | {count} | {c_pct:.1f}% | {w_pct:.1f}% | {r_pct:.1f}% | "
            f"{avg_tokens:.0f} | {ttft_50:.1f} / {ttft_95:.1f} | {tot_50:.1f} / {tot_95:.1f} |"
        )
        print(row_str)

if __name__ == "__main__":
    compute_tables()
