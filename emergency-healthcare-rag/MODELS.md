# Model Development Guide

This document describes how to create new models for the emergency healthcare RAG task.

## Shared Infrastructure

### Required Interface
All models must implement a `predict` function in `{model-name}/model.py`:

```python
def predict(statement: str) -> Tuple[int, int]:
    """
    Args:
        statement (str): Medical statement to classify
        
    Returns:
        Tuple[int, int]: (statement_is_true, statement_topic)
            - statement_is_true: 1 if true, 0 if false  
            - statement_topic: topic ID from 0-114
    """
```

### Model Selection
Change the active model in `/model.py`:
```python
ACTIVE_MODEL = "model-1"  # Change to your model directory name
```

### Shared Components
- **`api.py`** - FastAPI endpoint (calls `model.predict()`)
- **`utils.py`** - Validation and data loading utilities
- **`data/topics.json`** - Topic name to ID mapping (115 topics)
- **`data/topics/`** - Medical reference documents (organized by topic)
- **`data/train/`** - Training statements and answers
- **`evaluate_detailed.py`** - Comprehensive evaluation script
- **`example.py`** - Simple test script

### Data Format
- **Statements**: Plain text files (`statement_XXXX.txt`)
- **Answers**: JSON files with `statement_is_true` (0/1) and `statement_topic` (0-114)
- **Medical content**: Markdown files organized by topic directories

## Creating New Models

1. **Create model directory**: `model-X/` (replace X with your identifier)

2. **Required files**:
   - `model-X/model.py` - Main predict function (required interface)
   - `model-X/__init__.py` - Python module initialization (can be empty)

3. **Optional structure** (model-1 example):
   - `search.py` - Document retrieval/semantic search
   - `llm.py` - LLM integration
   - `generate_embeddings.py` - Preprocessing scripts
   - `embeddings.pkl` - Model-specific cached data

4. **Update model selection** in `/model.py`:
   ```python
   ACTIVE_MODEL = "model-X"
   ```

5. **Test your model**:
   ```bash
   python example.py                    # Quick test
   python evaluate_detailed.py 5       # Detailed evaluation
   python api.py                       # Start API server
   ```

## Constraints

- **Speed**: Max 5 seconds per prediction
- **Privacy**: No cloud API calls during inference  
- **Memory**: Max 24GB VRAM
- **Dependencies**: Add to `requirements.txt` if needed

## Model Independence

Each model directory is completely independent:
- Can use different approaches (RAG, fine-tuning, ensembles, etc.)
- Can have different dependencies and preprocessing
- Can store model-specific data (embeddings, weights, etc.)
- Should not modify shared components

The only requirement is implementing the `predict()` interface correctly.