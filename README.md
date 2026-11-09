# The Relay: Resilient Research QA Pipeline

A resilient, cited research paper question-answering pipeline and calibrated model stand-in evaluating Groq API constraints across three arXiv preprints:
- *Attention Is All You Need* (arXiv:1706.03762v7)
- *Moshi: a speech-text foundation model for real-time dialogue* (arXiv:2410.00037v1)
- *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (arXiv:2507.07955v2)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # Add your GROQ_API_KEY (optional; stand-in runs offline)
```

## Quickstart

```bash
# Run unit tests
pytest tests/ -v

# Run retrieval evaluation & 5 load scenarios (generates results.csv)
python scripts/run_benchmarks.py

# Recompute Tables 1, 2a, 2b, 3 from results.csv
python scripts/generate_tables.py
```

## Structure

- `src/`: Pipeline, retriever, Groq client, and calibrated stand-in service.
- `data/questions.json`: 16 curated evaluation questions.
- `results.csv`: 175 captured benchmark requests.
- `report/report.tex`: Complete Overleaf-ready LaTeX technical report.
- `submission/`: Clean submission deliverables for evaluation.

