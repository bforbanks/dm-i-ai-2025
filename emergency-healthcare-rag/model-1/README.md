# Model-1: RAG with Configurable Topic Selection

Retrieval-Augmented Generation system using semantic search and local LLM.

## Architecture

**Two-stage approach:**
1. **Topic Identification**: Semantic search through medical documents
2. **Truth Classification**: LLM evaluation with relevant context

## Components

### Document Processing (`search.py`)
- **Chunking**: 500 words per chunk, 50-word overlap
- **Embeddings**: Sentence-transformers (`all-MiniLM-L6-v2`)
- **Source**: 115 medical topics from `data/topics/` (markdown files)
- **Storage**: Pre-computed embeddings in `embeddings.pkl`

### LLM Integration (`llm.py`) 
- **Model**: Ollama with `gemma3:12b`
- **Truth classification**: Simple binary prompt (1/0 response)
- **Topic+Truth combined**: Single LLM call for both tasks

### Main Logic (`model.py`)
**Configurable via `TOPIC_CANDIDATES_K` (currently K=3):**

- **K=1**: Pure semantic topic selection + LLM truth classification
- **K>1**: LLM chooses from top-K semantic candidates + determines truth

## Current Configuration (K=3)

1. **Semantic search** finds top 3 topic candidates
2. **Context assembly** from multiple candidate topics (~1500 chars total) 
3. **Single LLM call** selects best topic AND determines truth
4. **Response parsing** extracts topic_id and truth_value

## Performance

- **Memory**: Embeddings ~10MB, fits VRAM constraint easily
- **Speed**: Meets 5-second constraint requirement
- **Accuracy**: Configurable K parameter allows speed/accuracy trade-offs

## Setup

```bash
# Setup local model and generate embeddings (one-time setup)
python model-1/setup_local_embeddings.py

# Test the model
python example.py

# Or test with detailed evaluation
python model-1/evaluate_detailed.py 5
```

## Approach Benefits

- **Offline**: No cloud API calls during inference
- **Context-aware**: Uses relevant medical literature  
- **Configurable**: Adjustable semantic vs LLM balance
- **Fast**: Single LLM call regardless of K value