# Combined Model 2 - Enhanced RAG with Hybrid Search

## Overview

Combined Model 2 is an enhanced version of the RAG system that uses hybrid search (BM25 + Vector) and a single LLM call approach for better accuracy and performance.

## Key Improvements

### 1. Hybrid Search
- **BM25**: For exact medical terminology matching
- **Vector Search**: For semantic similarity using `all-mpnet-base-v2`
- **Reciprocal Rank Fusion (RRF)**: Intelligent combination of results

### 2. Medical-Aware Chunking
- **Semantic chunking**: Split on medical headers (`##`) rather than arbitrary words
- **Larger chunks**: 600 words with 100-word overlap for better context
- **Medical concept preservation**: Keeps related medical information together

### 3. Single LLM Call Strategy
- **One call**: Both topic selection and truth classification in single inference
- **Rich context**: 2500 characters of targeted medical context
- **Enhanced prompts**: Medical domain-specific prompt engineering

## Architecture

```
Statement → Hybrid Search (BM25 + Vector) → Top-8 Candidates → 
Single LLM Call → Both Truth & Topic Classification
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install sentence-transformers rank-bm25
   ```

2. **Prepare data** (if not already done):
   ```bash
   python topics_processing/create_condensed_topics.py
   python topics_processing/clean_condensed_topics.py
   ```

3. **Setup model**:
   ```bash
   python combined-model-2/setup.py
   ```

## Usage

```python
from combined_model_2.model import predict

# Classify a medical statement
truth, topic = predict("Euglycemic diabetic ketoacidosis is characterized by blood glucose less than 250 mg/dL with metabolic acidosis.")
print(f"Truth: {truth}, Topic: {topic}")
```

## Performance Results

### Accuracy (10 Test Cases)
- **Truth Accuracy**: 100% (10/10)
- **Topic Accuracy**: 80% (8/10)
- **Overall Accuracy**: 90%

### Speed (M4 Mac - Development)
- **Average time per statement**: 9.55s
- **Statements per second**: 0.10
- **Model loading**: Cached for efficiency
- **Expected cloud performance**: Much faster with GPU acceleration

### Key Features

1. **Hybrid Search**: Combines keyword and semantic search for better retrieval
2. **Medical Chunking**: Preserves medical concept boundaries
3. **Single LLM Call**: Efficient inference with rich context
4. **Enhanced Prompts**: Medical domain-specific instructions
5. **Reciprocal Rank Fusion**: Optimal result combination
6. **Model Caching**: Avoids repeated model loading
7. **Increased Coverage**: 8 topic candidates for better accuracy

## Model Components

- **Embedding Model**: `all-mpnet-base-v2` (upgraded from MiniLM)
- **LLM**: `gemma3n:e4b` (as requested)
- **Search**: BM25 + Vector + RRF
- **Chunking**: Medical-aware semantic chunking

## Files

- `model.py`: Main prediction interface
- `search.py`: Hybrid search implementation
- `llm.py`: LLM classification logic
- `setup.py`: Setup and testing script
- `test_examples.py`: Accuracy testing
- `performance_test.py`: Performance benchmarking
- `embeddings.pkl`: Pre-computed embeddings and BM25 index

## Comparison with Combined-Model-1

| Metric | Combined-Model-1 | Combined-Model-2 |
|--------|------------------|------------------|
| Accuracy | 69% | 90% (10 test cases) |
| Speed | 1.5s | 9.55s (M4 Mac) |
| Search Method | Vector only | Hybrid (BM25 + Vector) |
| LLM Calls | 2 separate | 1 combined |
| Context Size | 1500 chars | 2500 chars |
| Candidates | 1 | 8 |

## Notes

- Performance optimized for accuracy over speed on development machine
- Expected dramatic speed improvement on cloud GPU instance
- Model loading is cached to avoid repeated initialization
- Uses medical-aware chunking for better context preservation
- Single LLM call reduces overhead and improves consistency
- Enhanced prompts with detailed medical guidelines
- Increased candidate coverage for better topic selection 