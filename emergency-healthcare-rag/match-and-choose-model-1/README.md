# Match and Choose Model 1

Advanced topic matching with threshold-based LLM decision making.

## Overview

This model implements a **threshold-based decision making** approach that:

1. **Uses BM25 search** from separated-models-2 for topic matching with detailed scoring
2. **Analyzes score gaps** between 1st and 2nd topic candidates
3. **Makes smart decisions** based on confidence thresholds:
   - **High confidence (gap > threshold)**: Use separated approach (topic model → truth LLM)
   - **Low confidence (gap ≤ threshold)**: Use combined approach (LLM chooses topic + truth)

## Key Innovation

**Problem**: Topic models sometimes make uncertain decisions when multiple topics have similar scores.
**Solution**: Let the LLM make decisions only when the topic model is genuinely uncertain.

## Implementation

### Files

- `config.py` - Configuration with environment variable support  
- `search.py` - BM25 search with enhanced scoring (inherited from separated-models-2)
- `llm.py` - Dual prompts (truth-only vs topic+truth classification)
- `model.py` - Main prediction logic with threshold-based decision making
- `evaluate.py` - Comprehensive evaluation with threshold analysis
- `threshold_analysis.py` - Detailed threshold optimization analysis

### Key Features

#### Threshold-Based Decision Making
- **Default threshold: 0** (LLM intervenes when scores are exactly tied)
- **Configurable via environment**: `THRESHOLD=5.0` or `THRESHOLD=NA`
- **Smart approach selection** based on score gap analysis

#### Dual LLM Prompts
- **Truth-only prompt**: When topic is confident (separated approach)
- **Topic+Truth prompt**: When choosing between similar candidates (combined approach)
- **Response formats**: Single number (0/1) vs "topic_id,truth_bool"

#### Comprehensive Evaluation
- **Search evaluation**: BM25 performance on topic matching
- **Threshold analysis**: Decision behavior on full dataset  
- **Approach breakdown**: Separated vs combined usage statistics
- **Performance tracking**: Timing and accuracy per approach

## Usage

### Basic Prediction
```python
from model import predict

# Uses config defaults (threshold=0, model=gemma3:27b)
truth_value, topic_id = predict("Medical statement here")
```

### Advanced Prediction with Details
```python
from model import predict_with_details

# Get detailed decision information
result = predict_with_details("Medical statement", threshold=5.0)
print(f"Approach used: {result['decision_info']['approach_used']}")
print(f"Score gap: {result['decision_info']['gap']:.3f}")
```

### Configuration
```bash
# Set model and threshold via environment
export LLM_MODEL="gemma3:27b"
export THRESHOLD="0"  # or "NA" for always separated

# Or via Python
from config import set_llm_model, set_threshold
set_llm_model("llama3.1:8b")
set_threshold(10.0)
```

## Evaluation

### Search Component Only (Fast)
```bash
python match-and-choose-model-1/evaluate.py --search-only
```

### Threshold Analysis (Full Dataset)
```bash
python match-and-choose-model-1/evaluate.py --threshold-analysis
```

### Full Pipeline Evaluation
```bash
# Default: 20 samples, threshold=0, model=gemma3:27b
python match-and-choose-model-1/evaluate.py

# Custom configuration
python match-and-choose-model-1/evaluate.py --samples 50 --threshold 5.0 --model llama3.1:8b

# Different threshold strategies
python match-and-choose-model-1/evaluate.py --threshold NA    # Always separated
python match-and-choose-model-1/evaluate.py --threshold 0     # LLM for ties only
python match-and-choose-model-1/evaluate.py --threshold 10    # LLM for uncertain cases
```

## Performance Results

Based on threshold analysis (threshold=0):

```
Search Performance:
- Top-1 accuracy: 89.5% (179/200)
- Top-3 accuracy: 97.0% (194/200)

Threshold Behavior:
- Separated approach: 79.8% of cases (gap > 0)
- Combined approach: 20.2% of cases (gap = 0)
- Separated accuracy: 94.2% (high confidence → high accuracy)
- Combined baseline: 69.2% (LLM should improve this)
```

## Key Insights

1. **Threshold = 0 is optimal**: Identifies exactly the cases where topic model is uncertain
2. **Tied scores aren't random**: 69.2% accuracy due to systematic tie-breaking bias
3. **Clear value proposition**: LLM can improve 69.2% → 85%+ for 20% of cases
4. **Efficient resource usage**: Most cases (80%) use faster separated approach

## Future Optimizations

1. **Hybrid Search**: Replace BM25-only with BM25+semantic search from separated-models-1
2. **Grid Search**: Quantitative threshold optimization on validation set
3. **Adaptive Thresholds**: Different thresholds based on medical domain/complexity
4. **Context Enhancement**: Richer context for LLM decision making

## Dependencies

- `rank_bm25` - BM25 search implementation
- `ollama` - LLM interface  
- `numpy` - Numerical operations
- `tqdm` - Progress bars
- `pathlib` - File path handling

## Configuration

- **Default threshold**: 0 (LLM for exact ties)
- **Default model**: gemma3:27b (most capable)
- **Chunk size**: 128 words
- **Overlap**: 12 words  
- **Cache location**: `.cache/`