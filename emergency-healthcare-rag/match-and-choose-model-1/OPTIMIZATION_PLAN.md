# Topic Model Optimization

## Current Performance
- BM25-only search: 89.5% top-1 accuracy
- Your recent result: 90% with `all-distilroberta-v1` + `linear_0.5`

## Goal
Find better hybrid BM25 + semantic search combinations and optimize BM25 parameters.

## Usage
```bash
cd emergency-healthcare-rag/
python match-and-choose-model-1/optimize_topic_model.py
```

## What It Tests
- 3 embedding models (MiniLM, MPNet, DistilRoBERTa)
- 7 fusion strategies (BM25-only, semantic-only, linear combinations, RRF, adaptive)
- Multiple BM25 chunk sizes and overlap ratios
- Condensed vs original topics

## Configuration
Edit lines ~499-504 in `optimize_topic_model.py`:
- `optimize_bm25=True` for full BM25 optimization (longer runtime)
- `use_condensed_topics=True` for condensed topics
- `max_samples=200` for evaluation size

## Expected Results
- Target: 91-93% accuracy (+1.5-3.5% improvement)
- Runtime: 60-300 minutes depending on config
- Output: `optimization_results_topic_model.json`