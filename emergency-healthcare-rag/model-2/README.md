# Model-2: Emergency Healthcare RAG - Next Iteration

## Overview

Model-2 is the next iteration of the emergency healthcare RAG system, building on the lessons learned from model-1.

## Goals for Model-2

- **Speed Optimization**: Target <5 second inference time
- **Improved Accuracy**: Better topic classification and truth detection
- **Offline Operation**: No dependency on online services during inference

## Current Status

🚧 **Under Development** 🚧

- [ ] Define architecture approach
- [ ] Implement core prediction logic
- [ ] Add evaluation and timing analysis
- [ ] Optimize for speed constraints
- [ ] Test on UCloud GPU instances

## Usage

```python
# Test model-2 locally
python model-2/evaluate_detailed.py 5

# Run via main interface (when ACTIVE_MODEL = "model-2")
python example.py
```

## Structure

```
model-2/
├── __init__.py           # Package initialization
├── model.py              # Main prediction interface
├── evaluate_detailed.py  # Evaluation and timing analysis
└── README.md            # This file
```

## Next Steps

1. **Architecture Decision**: Define the approach (RAG, fine-tuning, hybrid, etc.)
2. **Implementation**: Build the core prediction pipeline
3. **Speed Testing**: Validate <5s constraint on UCloud
4. **Accuracy Optimization**: Tune for better performance

## Notes

- Model-2 inherits the established interface: `predict(statement: str) -> Tuple[int, int]`
- Evaluation tooling is ready for testing
- Can fall back to model-1 components during development