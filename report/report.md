# The Relay: Technical Report

## 1. What I built and why
<!-- The design, and the one decision that shaped it most. -->

## 2. The real service and my stand-in
<!-- The limits you found in the real service and how you found them.
How your stand-in behaves, how you calibrated it from real measurements, and where it differs most from the real thing. -->

### Table 1: Limits you found in the real service
| Limit | Value you found | How you found it | Confidence (high, medium, low) |
| :--- | :--- | :--- | :--- |
| Requests per minute | | | |
| Tokens per minute | | | |
| Requests per day | | | |
| Tokens per day | | | |
| Requests in flight at once | | | |
| What it returns when a limit is hit | | | |
| Time until it recovers | | | |

## 3. Results
<!-- Time to first token and total time, each at the median and at the slow end (say which percentiles and how many requests), failures, and given-up requests, for each load shape. Say which runs were real and which used the stand-in. Put every number in the results tables below. Include a number we could not have guessed without running it. Also report answer quality on at least 15 questions you wrote yourself, and tokens used per answer. -->

### Table 2a: Latency
*(One row per load scenario: steady, burst, sustained overload, per-minute limit halved, daily budget exhausted)*
| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Steady | | | | | | |
| Burst | | | | | | |
| Sustained overload | | | | | | |
| Per-minute limit halved | | | | | | |
| Daily budget exhausted | | | | | | |

### Table 2b: Outcomes
*(Same rows as Table 2a)*
| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Steady | | | | | | | |
| Burst | | | | | | | |
| Sustained overload | | | | | | | |
| Per-minute limit halved | | | | | | | |
| Daily budget exhausted | | | | | | | |

## 4. Where the time goes
<!-- For one request that was slow, split its latency into waiting, working and retrying, and say whether the delay came before or after the first token. Show how you know. -->

## 5. When a limit is hit
<!-- What your system does at the per-minute limit and at the per-day limit. Who is delayed, who is dropped, who gets a weaker answer, and why. -->

## Retrieval: quality versus cost
<!-- Try at least three settings for how you split the papers and how much you fetch (piece size, number of pieces, and any re-ranking). For each, report answer quality, tokens per answer, and time to first token. Choose one setting and say why, and say what you do with retrieval when the limit is close. -->

### Table 3: Retrieval settings (at least three)
| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Setting A (200t, k=6, no re-rank) | | | | | | | |
| Setting B (500t, k=4, re-rank yes) | | | | | | | |
| Setting C (1000t, k=2, no re-rank) | | | | | | | |

## How this library will mislead a naive retriever
<!-- Name three ways, each with an example question where it goes wrong. -->

## 6. Retries
<!-- Your policy, and one case where retrying made things worse. -->

## 7. What the person sees
<!-- At 3pm the daily budget is gone. Describe the reply a waiting person receives. -->

## 8. What I'm unsure about, what I left out, what I learned
<!-- Your least certain assumption and what breaks if it is wrong. Two things you deliberately did not build. One thing that was new to you, taught to us in your own words. -->

## Your feedback on Vaani
<!-- Visit https://www.vaani.dev/ and tell us what you honestly think, in a few lines: what stood out, what confused you, and one thing you would change. There is no right answer. -->

## How I worked with the assistant
<!-- Include your real conversation history with it, not a cleaned-up one, or the prompts if the tool cannot export. Then, in half a page, give the three prompts that mattered most and why, one thing it got wrong that you caught and how you caught it, and one decision where you overruled it. Say what in this submission is distinctly yours. -->
