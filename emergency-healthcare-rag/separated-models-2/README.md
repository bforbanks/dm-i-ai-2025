# Separated Models 2 - BM25-Only RAG

This is an improved version of the emergency healthcare RAG system that uses **BM25-only search** with optimized chunking parameters based on Elias's research.

## Key Improvements

- **BM25-only search**: Removed semantic/vector search components for simplicity and speed
- **Optimized chunking**: Uses `chunk_size=128` and `overlap=12` based on fine-tuning results
- **Better performance**: Expected to match or exceed the previous hybrid approach
- **Faster inference**: No embedding generation required during search

## Performance Expectations

Based on Elias's results with the same configuration:
```
Top- 1: 0.895 (179/200)
Top- 3: 0.970 (194/200)
Top- 5: 0.975 (195/200)
Top-10: 0.985 (197/200)
```

## Architecture

```
Statement → BM25 Search → Topic ID → Context Retrieval → LLM → Response
```

### Components

1. **BM25 Search** (`search.py`): Optimized keyword-based search
2. **LLM Interface** (`llm.py`): Model-agnostic LLM integration
3. **RAG Pipeline** (`model.py`): Combines search and LLM
4. **Configuration** (`config.py`): Model selection and settings

## Usage

### Basic Usage

```python
from model import create_rag_model

# Create RAG model
rag_model = create_rag_model()

# Process a statement
result = rag_model.process_statement("Chest pain radiating to left arm indicates MI.")
print(result['response'])  # Returns 1 for TRUE, 0 for FALSE
```

### Running Examples

```bash
# From emergency-healthcare-rag/ directory
python separated-models-2/example.py
```

### Evaluation

```bash
# Test search functionality (no LLM required)
python separated-models-2/test_search.py

# Evaluate search component only
python separated-models-2/search.py

# Evaluate full pipeline (requires LLM)
python separated-models-2/evaluate.py
```

## Configuration

### LLM Models

Available models (configurable via `LLM_MODEL` environment variable):
- `gemma3n:e4b` (default) - Efficient 4B parameter model
- `llama3.1:8b` - Fast 8B parameter model
- `llama3.1:12b` - Balanced 12B parameter model
- `gemma3:27b` - Most capable 27B parameter model

### Search Parameters

- **Chunk Size**: 128 words
- **Overlap**: 12 words
- **Data Source**: `data/condensed_topics/`

## Files

- `search.py` - BM25 search implementation
- `llm.py` - LLM interface using Ollama
- `model.py` - Main RAG pipeline
- `config.py` - Configuration management
- `evaluate.py` - Evaluation script
- `example.py` - Usage examples

## Caching

BM25 index is cached in `.cache/bm25_index_condensed_128_12.pkl` for faster subsequent runs.

## Comparison with Separated Models 1

| Aspect | Separated Models 1 | Separated Models 2 |
|--------|-------------------|-------------------|
| Search Method | Hybrid (BM25 + Vector) | BM25-only |
| Chunk Size | 500 | 128 |
| Overlap | 150 | 12 |
| Speed | Slower (embeddings) | Faster (no embeddings) |
| Complexity | Higher | Lower |
| Expected Accuracy | 88.0% | ~89.5%+ |

## Requirements

- `rank_bm25` - For BM25 search
- `ollama` - For LLM interface
- `numpy` - For numerical operations
- `tqdm` - For progress bars 