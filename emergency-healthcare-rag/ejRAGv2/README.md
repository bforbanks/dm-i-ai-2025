# ejRAGv2: Hybrid Retrieval with Bayesian Optimization

This project implements a hybrid retrieval system that combines BM25 and dense embedding-based similarity (MiniLM or ColBERT) using Bayesian optimization to find optimal hyperparameters.

## Overview

The hybrid retrieval system uses a weighted linear combination of BM25 and embedding scores:
```
Hybrid Score = α × BM25_Score + β × Embedding_Score
```

Where α and β are learned weights optimized through Bayesian optimization.

## Features

- **Hybrid Retrieval**: Combines BM25 and dense embeddings for better performance
- **Bayesian Optimization**: Efficiently searches 9-dimensional hyperparameter space
- **Multiple Models**: Support for MiniLM and ColBERT embedding models
- **Comprehensive Analysis**: Built-in visualization and analysis tools
- **Results Storage**: All results saved in JSON format for reproducibility

## Hyperparameters Optimized

### BM25 Parameters
- `chunk_size_bm25`: Size of text chunks (64-128)
- `overlap_bm25`: Overlap between chunks (8-32)
- `k1`: BM25 term frequency saturation parameter (0.5-2.0)
- `b`: BM25 length normalization parameter (0.1-1.0)

### Embedding Parameters
- `chunk_size_embed`: Size of text chunks for embeddings (64-128)
- `overlap_embed`: Overlap between chunks (8-32)
- `model_selector`: Choice between MiniLM (0) and ColBERT (1)

### Fusion Parameters
- `alpha`: Weight for BM25 scores (0.0-1.0)
- `beta`: Weight for embedding scores (0.0-1.0)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure data is available in the correct structure:
```
data/
├── topics/           # Topic documents
├── train/
│   ├── statements/   # Training statements
│   └── answers/      # Ground truth answers
└── topics.json       # Topic mapping
```

## Usage

### 1. Test the Implementation

First, run the test suite to verify everything works:

```bash
python test_hybrid.py
```

### 2. Run Bayesian Optimization

Execute the main optimization script:

```bash
python hybrid_retrieval_bo.py
```

This will:
- Run 10 initial random points
- Perform 40 optimization iterations
- Save results to `hybrid_bo_results.json`

### 3. Analyze Results

Generate visualizations and analysis:

```bash
python analyze_results.py
```

This creates:
- `analysis_output/optimization_progress.png`: Optimization convergence
- `analysis_output/parameter_importance.png`: Parameter correlations
- `analysis_output/parameter_distributions.png`: Parameter distributions
- `analysis_output/optimization_summary.txt`: Detailed summary report

## File Structure

```
ejRAGv2/
├── hybrid_retrieval_bo.py    # Main optimization script
├── analyze_results.py        # Analysis and visualization
├── test_hybrid.py           # Test suite
├── requirements.txt         # Dependencies
├── README.md               # This file
├── hybrid_bo_results.json  # Optimization results (generated)
└── analysis_output/        # Analysis outputs (generated)
```

## Key Components

### Hybrid Retrieval Pipeline

1. **Text Chunking**: Documents are split into overlapping chunks
2. **BM25 Indexing**: Traditional keyword-based retrieval
3. **Embedding Indexing**: Dense vector representations
4. **Score Fusion**: Weighted combination of both retrieval methods
5. **Evaluation**: Top-1 accuracy on validation statements

### Bayesian Optimization

- Uses `bayes-opt` library for efficient hyperparameter search
- Explores 9-dimensional parameter space
- Maximizes top-1 accuracy objective
- Includes exploration vs exploitation balance

### Analysis Tools

- **Progress Tracking**: Monitor optimization convergence
- **Parameter Importance**: Identify most influential parameters
- **Distribution Analysis**: Understand successful vs unsuccessful configurations
- **Summary Reports**: Comprehensive performance analysis

## Configuration

You can modify the optimization parameters in `hybrid_retrieval_bo.py`:

```python
# Adjust search bounds
pbounds = {
    "chunk_size_bm25": (64, 128),
    "overlap_bm25": (8, 32),
    # ... other parameters
}

# Adjust optimization settings
optimizer = run_bayesian_optimization(init_points=10, n_iter=40)
```

## Results Interpretation

The optimization results include:

1. **Best Configuration**: Parameters achieving highest accuracy
2. **Parameter Importance**: Which parameters most affect performance
3. **Model Selection**: Whether MiniLM or ColBERT performs better
4. **Weight Analysis**: Optimal balance between BM25 and embeddings

## Performance Considerations

- **Memory**: Embedding models require significant RAM
- **Speed**: Each evaluation rebuilds indices (no caching for simplicity)
- **Scalability**: Can be extended with caching and parallel evaluation

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce chunk sizes or use smaller models
2. **Slow Performance**: Reduce number of iterations or use fewer statements
3. **Import Errors**: Ensure all dependencies are installed

### Debug Mode

Add debug prints in `evaluate_config()` to track progress:

```python
def evaluate_config(...):
    print(f"Evaluating: {chunk_size_bm25}, {k1}, {alpha}, {beta}")
    # ... rest of function
```

## Future Enhancements

- **Caching**: Cache indices to speed up evaluation
- **Parallel Processing**: Evaluate multiple configurations simultaneously
- **Advanced Fusion**: Try non-linear fusion methods
- **Cross-validation**: More robust evaluation strategy
- **Model Integration**: Add more embedding models (BERT, RoBERTa, etc.)

## Citation

If you use this implementation, please cite:

```bibtex
@misc{ejragv2_hybrid_retrieval,
  title={Hybrid Retrieval with Bayesian Optimization},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/ejragv2}
}
```
