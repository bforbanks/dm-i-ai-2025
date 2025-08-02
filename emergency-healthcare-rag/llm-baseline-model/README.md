# LLM-Only Baseline Model

A simple baseline model that uses only the LLM to predict truth and topic for medical statements.

## How it works

1. **Input**: A medical statement
2. **Process**: 
   - Loads all 115 topic titles from `data/topics.json`
   - Creates a prompt with the statement and all topic options
   - Sends to LLM for prediction
3. **Output**: Truth (0/1) and topic name

## Files

- `model.py` - The main model implementation
- `evaluate.py` - Simple evaluation script
- `README.md` - This file

## Usage

```bash
# Test the model
python model.py

# Evaluate on 20 examples
python evaluate.py 20

# Evaluate on all examples
python evaluate.py
```

## Expected Performance

- **Speed**: Very fast (just one LLM call)
- **Accuracy**: Likely lower than model-1/ since no semantic search
- **Purpose**: Baseline to compare against more sophisticated approaches

## Interface

The model follows the same interface as other models:
- `predict(statement: str) -> tuple[bool, str]`
- Returns (truth, topic_name)

This makes it compatible with the main `model.py` in the parent directory. 