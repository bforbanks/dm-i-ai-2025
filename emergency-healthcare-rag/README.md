# Emergency Healthcare RAG
![Kiku](../images/emergency-rag-banner.png)

## Overview

Emergency department medical statement evaluation system that determines:
- **Truth classification**: Is the statement true or false?
- **Topic classification**: Which medical topic does it concern? (115 topics)

## Architecture

### Match-and-Choose Model
Our primary approach uses **adaptive threshold-based decision making**:

1. **BM25 search** finds top medical topics with confidence scores
2. **Gap analysis**: `gap = 1st_score - 2nd_score`
3. **Smart routing**:
   - **High confidence** (`gap > threshold`): Topic model picks, LLM classifies truth
   - **Low confidence** (`gap ≤ threshold`): LLM chooses topic + classifies truth

**Benefits:**
- **Efficient**: 80% of cases use fast separated approach
- **Accurate**: LLM intervention improves uncertain cases from 69% to 85%+
- **Adaptive**: Automatically identifies when help is needed

### Model-Agnostic System
Support for multiple approaches:
- `match-and-choose-model-1` - Adaptive threshold-based (default)
- `separated-models-2` - BM25 + truth LLM
- `combined-model-2` - BM25 + combined topic+truth LLM

## Quickstart

```bash
# Run with default model
python api.py

# Run with specific model
python api.py --model match-and-choose-model-1

# Evaluate model
python match-and-choose-model-1/evaluate.py

# Optimize search parameters
python match-and-choose-model-1/optimize_search.py
```

## Search Optimization

Grid search optimization tests:
- **BM25 vs Hybrid** (BM25 + semantic embeddings)
- **Condensed vs Regular topics**
- **Chunk sizes** (96-384 words) and **overlap** (8-32 words)
- **Embedding models** and **fusion strategies**

Expected improvement: **88% → 92-95%** topic accuracy.

## Constraints

- **Speed**: ≤5 seconds per statement
- **Privacy**: Completely offline - no API calls
- **Memory**: ≤24GB VRAM usage

## Data

- **200 training statements** + **200 validation statements** + **749 evaluation statements**
- **115 medical topics** (stroke, cardiac arrest, trauma, diagnostics, etc.)
- **StatPearls reference articles** in `data/topics/`

## Example

**Input**: "Acute myocardial infarction presents with chest pain."

**Output**: `{"statement_is_true": 1, "statement_topic": 4}`

*Topic 4 = "Acute Myocardial Infarction"*