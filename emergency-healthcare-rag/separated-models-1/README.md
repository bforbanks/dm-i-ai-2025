# Separated Models 1

A clean separation approach to medical statement classification using Retrieval-Augmented Generation (RAG).

## Architecture

This model separates concerns into two distinct phases:

1. **Topic Selection**: Hybrid search determines the most relevant medical topic
2. **Truth Determination**: LLM uses rich context to determine if the statement is true/false

### Key Improvements

- **Clean Separation**: Search handles topic selection, LLM handles truth determination
- **Rich Context**: LLM receives context from multiple chunks across topics (your friend's approach)
- **Better Chunking**: Smaller chunks (500 chars) with more overlap (150 chars) for granular context
- **Context Quality**: Re-ranks chunks by relevance to the specific statement

## Components

### Search Module (`search.py`)
- **Hybrid Search**: BM25 + vector search with Reciprocal Rank Fusion (RRF)
- **Topic Selection**: Returns the topic ID of the highest-scoring chunk
- **Rich Context**: Retrieves chunks from chosen topic AND related topics
- **Medical-Aware Chunking**: Splits on medical headers with overlap

### LLM Module (`llm.py`)
- **Truth-Only Classification**: Focused solely on determining true/false
- **Rich Context Usage**: Uses retrieved medical context for evidence-based decisions
- **Model-Agnostic**: Works with any Ollama model
- **Skeptical Prompting**: Defaults to false unless context provides strong evidence

### Configuration (`config.py`)
- **Model Selection**: Easy switching between different LLM models
- **Environment Variables**: `LLM_MODEL` for runtime model selection
- **Model Registry**: Centralized model information and descriptions

## Usage

### Setup
```bash
# Generate embeddings and test the model
python separated-models-1/setup.py
```

### Evaluation
```bash
# Evaluate on 50 samples
python separated-models-1/evaluate.py
```

### Model Switching
```bash
# List available models
python separated-models-1/model_switcher.py list

# Switch to a different model
export LLM_MODEL=llama3.1:8b
python separated-models-1/evaluate.py
```

## Performance Expectations

### Topic Accuracy
- Should be high since search is working well in previous models
- Clean separation removes LLM confusion about topic selection

### Truth Accuracy
- Should improve significantly with rich context
- LLM can now make evidence-based decisions
- Skeptical prompting should reduce false positives

### Speed
- Single LLM call (vs. combined approach)
- Rich context may increase token count but should improve accuracy
- Expected to be faster than combined-model-2

## Comparison with Previous Models

| Model | Topic Selection | Truth Determination | Context Usage |
|-------|----------------|-------------------|---------------|
| combined-model-1 | LLM + candidates | LLM + candidates | None |
| combined-model-2 | LLM + candidates | LLM + candidates | None |
| separated-models-1 | **Search only** | **LLM + rich context** | **Multiple chunks** |

## Key Innovations

1. **Your Friend's Approach**: Context from multiple chunks, not just deduplicated topics
2. **Evidence-Based Truth**: LLM makes decisions based on actual medical text
3. **Clean Separation**: Each component has a single, clear responsibility
4. **Rich Context**: 3000+ characters of relevant medical information

## Expected Advantages

- **Higher Truth Accuracy**: LLM has actual evidence to work with
- **Better Topic Accuracy**: Search is already performing well
- **Faster Inference**: Single LLM call instead of complex reasoning
- **More Reliable**: Less prone to LLM hallucinations or biases 