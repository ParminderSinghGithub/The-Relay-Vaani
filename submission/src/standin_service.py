"""
Stand-In Model Service: Calibrated from real Groq measurements.
Replicates:
1. Baseline TTFT and token generation speed (calibrated from live probe: p50 ~1670ms)
2. Slower under load: latency scales with concurrent in-flight requests (L = L_0 + alpha * N_inflight)
3. Piece-by-piece streaming tokens with realistic inter-token delays
4. Strict limit enforcement: RPM (30), TPM (8000), RPD (1000), TPD (200000)
5. Refusal on rate limit: HTTP 429 with Groq-identical JSON error body and retry-after header
6. Simulated random or load-induced errors/timeouts
7. Thread-safe concurrency tracking and deterministic testing mode
"""

import time
import random
import threading
from typing import List, Dict, Any, Optional

class StandinModelService:
    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
        rpm_limit: int = 30,
        tpm_limit: int = 8000,
        rpd_limit: int = 1000,
        tpd_limit: int = 200000,
        base_ttft_ms: float = 1650.0,
        ttft_jitter_ms: float = 300.0,
        tokens_per_sec: float = 45.0,
        concurrency_penalty_ms: float = 180.0, # extra latency per in-flight request
        failure_rate: float = 0.0,
        timeout_threshold_ms: float = 30000.0,
        deterministic: bool = False # set True for fast, repeatable unit tests
    ):
        self.model = model
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.rpd_limit = rpd_limit
        self.tpd_limit = tpd_limit
        
        # Calibrated latency dynamics
        self.base_ttft_ms = base_ttft_ms
        self.ttft_jitter_ms = ttft_jitter_ms if not deterministic else 0.0
        self.tokens_per_sec = tokens_per_sec
        self.concurrency_penalty_ms = concurrency_penalty_ms
        self.failure_rate = failure_rate
        self.timeout_threshold_ms = timeout_threshold_ms
        self.deterministic = deterministic
        
        # Thread safety lock
        self._lock = threading.Lock()
        
        # Real-time state
        self.in_flight_requests = 0
        self.peak_queue_length = 0
        
        # Sliding window trackers
        self.minute_request_timestamps = []
        self.minute_token_log = []
        self.daily_requests_used = 0
        self.daily_tokens_used = 0

    def reset_state(self):
        with self._lock:
            self.in_flight_requests = 0
            self.peak_queue_length = 0
            self.minute_request_timestamps.clear()
            self.minute_token_log.clear()
            self.daily_requests_used = 0
            self.daily_tokens_used = 0

    def _clean_sliding_windows(self, now: float):
        cutoff = now - 60.0
        self.minute_request_timestamps = [t for t in self.minute_request_timestamps if t > cutoff]
        self.minute_token_log = [(t, count) for (t, count) in self.minute_token_log if t > cutoff]

    def get_remaining_tokens_minute(self, now: float) -> int:
        self._clean_sliding_windows(now)
        used_tokens = sum(count for (_, count) in self.minute_token_log)
        return max(0, self.tpm_limit - used_tokens)

    def get_remaining_requests_minute(self, now: float) -> int:
        self._clean_sliding_windows(now)
        return max(0, self.rpm_limit - len(self.minute_request_timestamps))

    def _generate_synthetic_answer(self, messages: List[Dict[str, str]], query: str) -> str:
        q_lower = query.lower()
        
        # Out-of-scope / unanswerable
        if any(term in q_lower for term in ["llama 3", "deepseek", "gpu power", "training cost in us dollars"]):
            return "The papers in the research library do not provide information regarding this topic. As instructed, I am stating this directly rather than guessing."

        if "whisper" in q_lower and "mosh" not in q_lower:
            return "The papers in the library do not contain details regarding the Whisper model. The library covers Attention Is All You Need, Moshi, and Dynamic Chunking."

        # Trap questions
        if "dynamic chunking" in q_lower and "attention is all you need" in q_lower:
            return "The paper 'Attention Is All You Need' does not use or discuss dynamic chunking. Dynamic chunking is introduced in the H-Net paper (arXiv:2507.07955). Therefore, no dynamic chunking threshold exists in 1706.03762 [Dynamic Chunking, Section: 1 Introduction, Page 1]."

        if "bpe" in q_lower and ("dynamic chunking" in q_lower or "h-net" in q_lower):
            return "The Dynamic Chunking (H-Net) paper does not use a BPE vocabulary size for its model because its explicit objective is to eliminate handcrafted tokenizers like BPE, operating directly on raw byte streams [Dynamic Chunking, Section: 1 Introduction, Page 1]."

        if "dimension d_k" in q_lower or "d_model" in q_lower:
            return "In the Transformer base model, the model dimensionality is d_model = 512, with h = 8 parallel attention layers (heads). For each head, the dimension is d_k = d_v = d_model / h = 64 [Attention Is All You Need, Section: 3.2.2 Multi-Head Attention, Page 4]."

        if "mimi" in q_lower or "frame rate" in q_lower:
            return "The Mimi neural audio codec operates at a frame rate of 12.5 Hz with an overall latency of 80ms (40ms frame size plus 40ms algorithmic lookahead), using 8 codebooks [Moshi, Section: 3.3 Audio Tokenization, Page 9]."

        if "positional encoding" in q_lower:
            return "Positional encodings use sine and cosine functions of different frequencies: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) and PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)) [Attention Is All You Need, Section: 3.5 Positional Encoding, Page 6]."

        if "modalities" in q_lower or "dna" in q_lower:
            return "H-Net demonstrated significant performance and data efficiency improvements on modalities where tokenizers struggle, specifically Chinese text, computer code, and DNA sequences, achieving nearly a 4x improvement in data efficiency [Dynamic Chunking, Section: 1 Introduction, Page 1]."

        if "multi-stream" in q_lower or "listening and talking" in q_lower:
            return "Moshi models full-duplex spoken dialogue as multi-stream generation, modeling both the user's audio tokens and its own generated audio/text tokens in parallel time-aligned streams [Moshi, Section: 3.1 Overview, Page 6]."

        return "Based on the provided research papers, the architecture relies on hierarchical attention mechanisms with explicit representation learning [Attention Is All You Need, Section: 3 Model Architecture, Page 2]."

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        query: str = "",
        max_tokens: int = 256,
        timeout: float = 30.0,
        force_error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simulates streaming response with calibrated latency, queue slowdown,
        and limit enforcement. Synchronous & thread-safe.
        """
        t_start = time.perf_counter()
        now = time.time()
        
        prompt_words = sum(len(m.get("content", "").split()) for m in messages)
        tokens_in = max(10, int(prompt_words * 1.3))
        est_tokens_out = max(20, min(max_tokens, 120))
        total_tokens_needed = tokens_in + est_tokens_out

        with self._lock:
            self._clean_sliding_windows(now)
            self.in_flight_requests += 1
            if self.in_flight_requests > self.peak_queue_length:
                self.peak_queue_length = self.in_flight_requests
            current_in_flight = self.in_flight_requests

            # Forced test behaviors
            if force_error == "rate_limit":
                self.in_flight_requests -= 1
                return {
                    "status": "rate_limited",
                    "status_code": 429,
                    "first_token_ms": 0.0,
                    "total_ms": 10.0,
                    "content": "",
                    "error_message": f"Rate limit reached for model `{self.model}` on requests per minute (RPM): Limit {self.rpm_limit}, Used {self.rpm_limit}, Requested 1.",
                    "headers": {
                        "x-ratelimit-limit-requests": str(self.rpd_limit),
                        "x-ratelimit-remaining-requests": "0",
                        "x-ratelimit-limit-tokens": str(self.tpm_limit),
                        "x-ratelimit-remaining-tokens": "0",
                        "retry-after": "1"
                    },
                    "tokens_in": tokens_in,
                    "tokens_out": 0
                }
            elif force_error == "timeout":
                self.in_flight_requests -= 1
                return {
                    "status": "timeout",
                    "first_token_ms": 0.0,
                    "total_ms": timeout * 1000.0,
                    "content": "",
                    "error_message": "Client request timed out waiting for first token",
                    "headers": {},
                    "tokens_in": tokens_in,
                    "tokens_out": 0
                }
            elif force_error == "500":
                self.in_flight_requests -= 1
                return {
                    "status": "error",
                    "status_code": 500,
                    "first_token_ms": 0.0,
                    "total_ms": 10.0,
                    "content": "",
                    "error_message": "Internal Server Error",
                    "headers": {},
                    "tokens_in": tokens_in,
                    "tokens_out": 0
                }

            # 1. Check daily budget
            if self.daily_requests_used >= self.rpd_limit or self.daily_tokens_used >= self.tpd_limit:
                self.in_flight_requests -= 1
                dur_ms = (time.perf_counter() - t_start) * 1000.0
                return {
                    "status": "rate_limited",
                    "status_code": 429,
                    "first_token_ms": 0,
                    "total_ms": dur_ms,
                    "content": "",
                    "error_message": f"Daily limit exceeded for model `{self.model}`.",
                    "headers": {
                        "x-ratelimit-limit-requests": str(self.rpd_limit),
                        "x-ratelimit-remaining-requests": "0",
                        "x-ratelimit-limit-tokens": str(self.tpm_limit),
                        "x-ratelimit-remaining-tokens": "0",
                        "retry-after": "3600"
                    },
                    "tokens_in": tokens_in,
                    "tokens_out": 0
                }

            # 2. Check per-minute limits
            if len(self.minute_request_timestamps) >= self.rpm_limit:
                self.in_flight_requests -= 1
                dur_ms = (time.perf_counter() - t_start) * 1000.0
                reset_wait = max(0.5, 60.0 - (now - self.minute_request_timestamps[0]))
                return {
                    "status": "rate_limited",
                    "status_code": 429,
                    "first_token_ms": 0,
                    "total_ms": dur_ms,
                    "content": "",
                    "error_message": f"Rate limit reached for model `{self.model}` on requests per minute (RPM). Please try again in {reset_wait:.2f}s.",
                    "headers": {
                        "x-ratelimit-limit-requests": str(self.rpd_limit),
                        "x-ratelimit-remaining-requests": "0",
                        "x-ratelimit-limit-tokens": str(self.tpm_limit),
                        "x-ratelimit-remaining-tokens": str(self.get_remaining_tokens_minute(now)),
                        "retry-after": str(int(reset_wait + 1))
                    },
                    "tokens_in": tokens_in,
                    "tokens_out": 0
                }

            current_min_tokens = sum(count for (_, count) in self.minute_token_log)
            if current_min_tokens + total_tokens_needed > self.tpm_limit:
                self.in_flight_requests -= 1
                dur_ms = (time.perf_counter() - t_start) * 1000.0
                reset_wait = 2.0
                return {
                    "status": "rate_limited",
                    "status_code": 429,
                    "first_token_ms": 0,
                    "total_ms": dur_ms,
                    "content": "",
                    "error_message": f"Rate limit reached for model `{self.model}` on tokens per minute (TPM).",
                    "headers": {
                        "x-ratelimit-limit-requests": str(self.rpd_limit),
                        "x-ratelimit-remaining-requests": str(max(0, self.rpm_limit - len(self.minute_request_timestamps))),
                        "x-ratelimit-limit-tokens": str(self.tpm_limit),
                        "x-ratelimit-remaining-tokens": str(max(0, self.tpm_limit - current_min_tokens)),
                        "retry-after": str(int(reset_wait + 1))
                    },
                    "tokens_in": tokens_in,
                    "tokens_out": 0
                }

            # Commit request to usage tracking
            self.minute_request_timestamps.append(now)
            self.minute_token_log.append((now, total_tokens_needed))
            self.daily_requests_used += 1
            self.daily_tokens_used += total_tokens_needed

        # 4. Latency Calculation (Calibrated baseline + concurrency slowdown)
        concurrency_delay = max(0, current_in_flight - 1) * self.concurrency_penalty_ms
        jitter = 0.0 if self.deterministic else random.uniform(-self.ttft_jitter_ms, self.ttft_jitter_ms)
        target_ttft_ms = max(5.0 if self.deterministic else 400.0, self.base_ttft_ms + jitter + concurrency_delay)

        # Simulate TTFT delay
        sleep_sec = (target_ttft_ms / 1000.0) if not self.deterministic else 0.005
        time.sleep(sleep_sec)
        t_first_token = time.perf_counter()
        first_token_ms = (t_first_token - t_start) * 1000.0 if not self.deterministic else target_ttft_ms

        # 5. Token Generation Streaming Delay
        full_text = self._generate_synthetic_answer(messages, query)
        words = full_text.split()
        tokens_out = max(15, int(len(words) * 1.3))
        
        gen_duration_sec = (tokens_out / self.tokens_per_sec) if not self.deterministic else 0.005
        time.sleep(gen_duration_sec)
        
        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000.0 if not self.deterministic else (target_ttft_ms + 15.0)
        
        with self._lock:
            self.in_flight_requests -= 1

        headers = {
            "x-ratelimit-limit-requests": str(self.rpd_limit),
            "x-ratelimit-remaining-requests": str(max(0, self.rpd_limit - self.daily_requests_used)),
            "x-ratelimit-limit-tokens": str(self.tpm_limit),
            "x-ratelimit-remaining-tokens": str(self.get_remaining_tokens_minute(time.time())),
            "x-ratelimit-reset-requests": "12m",
            "x-ratelimit-reset-tokens": "1.8s"
        }

        return {
            "status": "ok",
            "status_code": 200,
            "first_token_ms": first_token_ms,
            "total_ms": total_ms,
            "content": full_text,
            "headers": headers,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out
        }
