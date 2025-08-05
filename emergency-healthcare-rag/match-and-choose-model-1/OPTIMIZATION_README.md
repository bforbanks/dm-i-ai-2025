# Topic Model Optimization Scripts

This directory contains optimization scripts for improving the match-and-choose topic model through hybrid BM25 + semantic search and advanced threshold optimization.

## Quick Start

### 1. Install Dependencies
```bash
pip install sentence-transformers scikit-learn matplotlib torch tqdm
```

### 2. Run Optimization (from `emergency-healthcare-rag/` directory)
```bash
# Full hybrid search optimization (2-4 hours)
python match-and-choose-model-1/optimize_topic_model.py

# Threshold analysis and failure case study (30-60 minutes)  
python match-and-choose-model-1/evaluate_threshold_optimization.py
```

## What Each Script Does

### `optimize_topic_model.py`
- Tests 3 semantic embedding models + BM25 combinations
- Evaluates 7 different fusion strategies (linear, RRF, adaptive)
- Downloads models automatically (no API calls)
- Generates comprehensive results with implementation recommendations
- **Output**: `optimization_results_topic_model.json`

### `evaluate_threshold_optimization.py`
- Analyzes current BM25 search performance patterns
- Simulates threshold-based LLM intervention strategies
- Identifies failure cases and recovery potential
- Provides optimal threshold recommendations
- **Output**: `threshold_optimization_results.json` + visualizations

## Expected Results

### Performance Improvements
- **Conservative**: +2-5% top-1 accuracy improvement
- **Optimistic**: +6-9% improvement with good complementary models
- **Current baseline**: 89.5% → Target: 92-95%+

### Key Insights
- Which semantic models complement BM25 best
- Optimal fusion strategies for different query types
- When to use LLM intervention vs fast topic model
- Failure pattern analysis for future improvements

## Files Generated

### Results Files
- `optimization_results_topic_model.json` - Complete hybrid search results
- `threshold_optimization_results.json` - Threshold analysis results
- `threshold_optimization_analysis.png` - Visualization plots

### Configuration Updates
Based on results, update:
- `search.py` - Add best hybrid search method
- `config.py` - Set optimal threshold values
- `model.py` - Integrate improved decision logic

## Understanding the Results

### Top Configuration Format
```json
{
  "model": "sentence-transformers/all-mpnet-base-v2",
  "strategy": "linear_0.7", 
  "top1_accuracy": 0.934,
  "top3_accuracy": 0.985,
  "mrr": 0.956,
  "time_per_query": 0.142
}
```

### Threshold Recommendations
```json
{
  "best_accuracy": {"threshold": 2.5, "topic_accuracy": 0.923},
  "best_balanced": {"threshold": 5.0, "separated_ratio": 0.65},
  "best_improvement": {"threshold": 1.0, "improvement": 0.034}
}
```

## Implementation Guide

### 1. Choose Best Hybrid Configuration
Look for the top-ranked configuration in `optimization_results_topic_model.json`:
- Balance accuracy vs speed
- Consider model download size for deployment
- Evaluate improvement over BM25-only baseline

### 2. Set Optimal Threshold
Use recommendations from `threshold_optimization_results.json`:
- **Conservative**: Use higher threshold (less LLM intervention)
- **Aggressive**: Use lower threshold (more LLM intervention)
- **Balanced**: Use median recommendation

### 3. Update Search Implementation
```python
# Example integration in search.py
from sentence_transformers import SentenceTransformer

class HybridSearcher:
    def __init__(self):
        self.bm25 = load_bm25_index()
        self.semantic_model = SentenceTransformer('best-model-from-results')
    
    def search(self, query, fusion_strategy='linear_0.7'):
        # Implement best strategy from optimization results
        pass
```

## Troubleshooting

### Common Issues
- **Model download failures**: Check internet connection, try different models
- **Memory errors**: Reduce batch size or use smaller embedding models
- **Slow execution**: Reduce `max_samples` in configuration
- **Import errors**: Ensure all dependencies are installed

### Performance Tips
- Use caching (enabled by default) for faster re-runs
- Start with smaller sample sizes for quick testing
- Run on machine with sufficient RAM (8GB+ recommended)

## Expected Outputs Summary

After running both scripts, you should have:
1. **Clear winner**: Best hybrid search configuration
2. **Optimal threshold**: Data-driven threshold recommendation  
3. **Improvement estimate**: Quantified accuracy gains
4. **Implementation plan**: Step-by-step integration guide
5. **Failure analysis**: Understanding of remaining limitations

This systematic optimization approach ensures data-driven improvements to the topic model with clear implementation guidance.