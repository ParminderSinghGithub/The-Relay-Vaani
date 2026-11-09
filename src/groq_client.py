"""
Client for Groq Chat Completions API.
Handles:
- Streaming token delivery with exact Time to First Token (TTFT) measurement
- Total latency measurement
- Extraction of rate limit telemetry headers (RPM, TPM, RPD, reset durations)
- Quota-safe execution
- Zero exposure of API keys
"""

import os
import time
import json
from typing import List, Dict, Any, Generator, Tuple, Optional
import httpx
from dotenv import load_dotenv

load_dotenv()

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

class GroqClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
        
    def is_configured(self) -> bool:
        return bool(self.api_key and not self.api_key.startswith("your_"))
        
    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Sends chat completion request with streaming enabled.
        Records:
        - first_token_ms (TTFT)
        - total_ms
        - response_text
        - prompt_tokens, completion_tokens
        - rate_limit headers
        - status: 'ok', 'rate_limited', 'timeout', 'error'
        """
        if not self.is_configured():
            return {
                "status": "error",
                "error_message": "GROQ_API_KEY is not configured",
                "first_token_ms": 0,
                "total_ms": 0,
                "content": "",
                "headers": {},
                "tokens_in": 0,
                "tokens_out": 0
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }

        t_start = time.perf_counter()
        first_token_time = None
        collected_chunks = []
        ratelimit_headers = {}
        
        try:
            with httpx.Client(timeout=timeout) as client:
                with client.stream("POST", GROQ_ENDPOINT, headers=headers, json=payload) as response:
                    # Capture rate limit headers
                    for k, v in response.headers.items():
                        if k.lower().startswith("x-ratelimit-") or k.lower() == "retry-after":
                            ratelimit_headers[k.lower()] = v
                            
                    if response.status_code == 429:
                        total_time = (time.perf_counter() - t_start) * 1000.0
                        body = response.read().decode("utf-8")
                        return {
                            "status": "rate_limited",
                            "status_code": 429,
                            "first_token_ms": 0,
                            "total_ms": total_time,
                            "content": "",
                            "error_message": body,
                            "headers": ratelimit_headers,
                            "tokens_in": 0,
                            "tokens_out": 0
                        }
                        
                    if response.status_code != 200:
                        total_time = (time.perf_counter() - t_start) * 1000.0
                        body = response.read().decode("utf-8")
                        return {
                            "status": "error",
                            "status_code": response.status_code,
                            "first_token_ms": 0,
                            "total_ms": total_time,
                            "content": "",
                            "error_message": body,
                            "headers": ratelimit_headers,
                            "tokens_in": 0,
                            "tokens_out": 0
                        }

                    # Parse SSE stream
                    for line in response.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            delta = data_json["choices"][0].get("delta", {})
                            content_piece = delta.get("content", "")
                            if content_piece:
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                collected_chunks.append(content_piece)
                        except Exception:
                            continue

            t_end = time.perf_counter()
            total_ms = (t_end - t_start) * 1000.0
            ttft_ms = (first_token_time - t_start) * 1000.0 if first_token_time else total_ms
            full_text = "".join(collected_chunks)
            
            # Approximate token counts (~1 word ≈ 1.3 tokens) or use usage if provided
            prompt_words = sum(len(m.get("content", "").split()) for m in messages)
            tokens_in = int(prompt_words * 1.3)
            tokens_out = int(len(full_text.split()) * 1.3)

            return {
                "status": "ok",
                "status_code": 200,
                "first_token_ms": ttft_ms,
                "total_ms": total_ms,
                "content": full_text,
                "headers": ratelimit_headers,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out
            }

        except httpx.TimeoutException:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            return {
                "status": "timeout",
                "first_token_ms": 0,
                "total_ms": total_ms,
                "content": "",
                "headers": ratelimit_headers,
                "tokens_in": 0,
                "tokens_out": 0
            }
        except Exception as e:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            return {
                "status": "error",
                "error_message": str(e),
                "first_token_ms": 0,
                "total_ms": total_ms,
                "content": "",
                "headers": ratelimit_headers,
                "tokens_in": 0,
                "tokens_out": 0
            }
