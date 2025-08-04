# Match and Choose Model 1

Advanced topic matching with LLM-based choice when matches are close.

## Overview

This model implements an enhanced search-based approach that:

1. **Uses BM25 search** from separated-models-2 for topic matching
2. **Provides detailed scoring** to determine when matches are "close"
3. **Will use LLM choice** when multiple topics have similar scores (future implementation)

## Current Implementation

### Files

- `search.py` - BM25 search with enhanced scoring and topic deduplication
- `evaluate_search.py` - Enhanced evaluation providing detailed topic matching analysis
- `test_search.py` - Simple test script for search functionality

### Key Features

#### Search Algorithm
- **BM25-only search** with optimized chunking (chunk_size=128, overlap=12)
- **Topic deduplication** - groups chunks by topic and keeps best score per topic
- **Enhanced scoring** - provides detailed match scores for analysis
- **Caching** - uses cached BM25 index for performance

#### Evaluation
- **Detailed rank analysis** - shows how often correct topic appears in top 1-10
- **Score analysis** - average scores for correct topics at each rank
- **Performance metrics** - timing and accuracy statistics
- **Comprehensive reporting** - saves detailed results to JSON

### Usage

#### Evaluate Topic Search
```bash
# Default: 200 samples, gemma3:27b model
python match-and-choose-model-1/evaluate_search.py

# Custom number of samples
python match-and-choose-model-1/evaluate_search.py --samples 100
```

#### Test Search Functionality
```bash
python match-and-choose-model-1/test_search.py
```

### Output Format

The evaluation provides:

1. **Per-sample analysis** showing:
   - Expected topic rank and score
   - Top 5 matches with scores
   - Search timing

2. **Overall statistics** including:
   - Rank analysis (top 1-10)
   - Percentage of correct topics at each rank
   - Average scores for correct topics
   - Timing statistics

3. **JSON results** saved to `search_evaluation_results.json`

### Example Output

```
Rank Analysis:
------------------------------------------------------------
Rank | Count | % | Cum.% | Avg Score
------------------------------------------------------------
   1 |    45 | 22.5% | 22.5% | 156.234
   2 |    38 | 19.0% | 41.5% | 142.891
   3 |    25 | 12.5% | 54.0% | 128.456
   4 |    18 |  9.0% | 63.0% | 115.234
   5 |    12 |  6.0% | 69.0% | 104.567
```

## Future Implementation

Once the search evaluation provides sufficient data, the model will implement:

1. **Score threshold logic** - determine when matches are "close"
2. **LLM choice mechanism** - when multiple topics have similar scores
3. **Combined topic/truth prediction** - format: "topic_id,is_truth"
4. **Full pipeline evaluation** - end-to-end testing

## Dependencies

- `rank_bm25` - BM25 search implementation
- `numpy` - numerical operations
- `tqdm` - progress bars
- `pathlib` - file path handling

## Configuration

- **Chunk size**: 128 words
- **Overlap**: 12 words  
- **Cache location**: `.cache/`
- **Default samples**: 200
- **Default model**: gemma3:27b 