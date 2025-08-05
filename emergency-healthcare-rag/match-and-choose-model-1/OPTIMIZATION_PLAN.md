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
**Default: Full Exploration with Adaptive Discovery**
```python
fast_mode=False                      # Enable full exploration 
top_bm25_configs=10                  # Test top 10 BM25 configs
sample_strategies=False              # Use all strategies
target_runtime_hours=8.0             # Allow longer runtime
enable_adaptive_exploration=True     # Detect breakthroughs and explore deeper
breakthrough_threshold=0.905         # Trigger exploration above 90.5%
exploration_radius=2                 # ±2 parameter variations
max_exploration_configs=50           # Up to 50 exploration configs per breakthrough
```

**For faster testing, set `fast_mode=True` and reduce `target_runtime_hours=1.5`**

## Expected Results
- **Phase 1**: Find 90-92% BM25-only configs (~20 minutes)
- **Phase 2**: Test semantic fusion on promising foundations (~4-6 hours)
- **Phase 3**: Adaptive exploration around breakthroughs (dynamic duration)
- **Target**: 91%+ accuracy (like your `all-distilroberta-v1 + linear_0.7` result)
- **Smart strategy**: Full exploration with adaptive focusing on promising paths

## Adaptive Exploration Features
**NEW**: When the script finds promising results (>90.5% accuracy), it automatically:

### **🔍 Breakthrough Detection**
- Monitors all results for accuracy above threshold
- Identifies configurations worth exploring deeper
- Example: Your `all-distilroberta-v1 + linear_0.7` at 91%

### **🚀 Smart Exploration Around Breakthroughs**
1. **BM25 Parameter Exploration**: Test chunk sizes ±32 and overlap ratios ±10% around successful configs
2. **Fusion Strategy Exploration**: If `linear_0.7` works, test `linear_0.5`, `linear_0.6`, `linear_0.8`, `linear_0.9`
3. **Model Architecture Exploration**: If DistilRoBERTa works, test related RoBERTa variants
4. **Recursive Exploration**: If exploration finds even better results, explore those too

### **🎯 Focus Computational Budget**
- Skip exhaustive testing on poor-performing areas
- Concentrate compute time on promising parameter regions
- Can run for hours focusing on breakthrough optimization

## Progress Tracking Features
Since ucloud terminal doesn't allow scrolling:
- **Phase markers**: Clear PHASE 1/PHASE 2/EXPLORATION separation
- **BM25 ranking**: See top performers after Phase 1
- **Breakthrough alerts**: Real-time notifications when breakthroughs found
- **Exploration progress**: Track adaptive exploration configs
- **Overall progress**: Real-time completion tracking with exploration counts