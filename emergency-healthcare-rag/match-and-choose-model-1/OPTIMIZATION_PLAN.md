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
### **🚀 SMART HIERARCHICAL OPTIMIZATION (94% faster!)**

### **Phase 1: DEEP BM25 Optimization** (~13 minutes)
- **Chunk sizes**: 64, 80, 96, 112, 128, 144, 160, 176, 192 (9 values)
- **Overlap ratios**: 0%, 5%, 10%, 15%, 20%, 25%, 30% (7 values)  
- **Total**: 63 BM25-only configurations
- **Goal**: Find optimal BM25 parameters, rank by performance

### **Phase 2: Smart Semantic Fusion** (~40 minutes)
- **Smart selection**: Only TOP 3 BM25 configs (vs all 63)
- **Models**: 3 embedding models (MiniLM, MPNet, DistilRoBERTa)
- **Strategies**: 4 best strategies (bm25_only, semantic_only, linear_0.5, rrf)
- **Total**: 27 semantic fusion configs (vs 540 exhaustive)

## Configuration
**Default: Fast Mode Enabled** (1 hour total runtime)
```python
fast_mode=True              # Enable hierarchical optimization
top_bm25_configs=3          # Test top 3 BM25 configs only  
sample_strategies=True      # Use 4 best strategies
target_runtime_hours=1.5    # Target runtime
```

## Expected Results
- **Phase 1**: Find 90-92% BM25-only configs (13 minutes)
- **Phase 2**: Test semantic fusion on best foundations (40 minutes)
- **Final Target**: 93-95% top-1 accuracy in **~1 hour total**
- **Efficiency**: 94% faster than exhaustive (1 hour vs 14+ hours)
- **Smart strategy**: Focus compute on promising configurations

## Progress Tracking Features
Since ucloud terminal doesn't allow scrolling:
- **Phase markers**: Clear PHASE 1/PHASE 2 separation
- **BM25 ranking**: See top performers after Phase 1
- **Smart selection**: Know which configs are tested in Phase 2
- **Overall progress**: Real-time completion tracking
- **Trend analysis**: Complete insights for next optimization