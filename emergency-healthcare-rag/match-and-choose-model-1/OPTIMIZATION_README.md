# Optimization Scripts

## Fixed Issues
- ✅ AttributeError resolved
- ✅ Warnings suppressed  
- ✅ BM25 optimization added
- ✅ Topic selection (condensed vs original)

## Quick Start
```bash
cd emergency-healthcare-rag/
python match-and-choose-model-1/optimize_topic_model.py
```

## Configuration
Edit lines ~499-504 for settings:
- BM25 optimization on/off
- Condensed vs original topics
- Sample size

## Output
- `optimization_results_topic_model.json` - Results ranked by accuracy
- Shows best model + strategy combinations
- Implementation recommendations