# Match-and-Choose Model Optimization

## Overview

This module implements a hierarchical optimization approach for the topic model component of the match-and-choose system. The optimization follows a 4-phase strategy to efficiently find the best hybrid BM25 + semantic search configuration.

## Optimization Strategy

### Phase 1: Extensive BM25 Optimization
- Tests 11 chunk sizes (64-224) × 8 overlap ratios (0.0-0.35) = 88 configurations
- Focuses purely on BM25 performance to find optimal text chunking parameters
- Fast execution since BM25 is computationally lightweight

### Phase 2: Embedding Model Optimization  
- Tests 5 embedding models × 5 chunk sizes × 5 overlap ratios = 125 configurations
- Downloads models locally (no online API calls)
- Finds best embedding model with its optimal hyperparameters
- Models tested:
  - `sentence-transformers/all-MiniLM-L6-v2` (fast, lightweight)
  - `sentence-transformers/all-mpnet-base-v2` (best general performance)
  - `sentence-transformers/all-distilroberta-v1` (different architecture)
  - `sentence-transformers/all-roberta-large-v1` (high capacity)
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual)

### Phase 3: Hybrid Combinations
- Takes top 5 BM25 configs × top 5 embedding configs × 6 fusion strategies = 150 combinations
- Tests different fusion strategies:
  - `bm25_only`: Pure BM25 baseline
  - `semantic_only`: Pure semantic baseline  
  - `linear_0.3`: 30% BM25 + 70% semantic
  - `linear_0.5`: Balanced fusion
  - `linear_0.7`: 70% BM25 + 30% semantic
  - `rrf`: Reciprocal Rank Fusion

### Phase 4: Zoom into Promising Configurations
- Takes top 3 most promising hybrid configurations
- Generates parameter variations around each (radius=2)
- Fine-tunes promising configurations for maximum performance

## Usage

```bash
cd emergency-healthcare-rag/match-and-choose-model-1/
python optimize_topic_model.py
```

## Configuration

The optimization can be customized via the `OptimizationConfig` class:

```python
@dataclass
class OptimizationConfig:
    # Phase 1: BM25 optimization
    bm25_chunk_sizes: List[int] = [64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224]
    bm25_overlap_ratios: List[float] = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    
    # Phase 2: Embedding optimization
    embedding_models: List[str] = [...]  # 5 models
    embedding_chunk_sizes: List[int] = [96, 112, 128, 144, 160]
    embedding_overlap_ratios: List[float] = [0.05, 0.1, 0.15, 0.2, 0.25]
    
    # Phase 3: Combination parameters
    top_bm25_configs: int = 5
    top_embedding_configs: int = 5
    fusion_strategies: List[str] = [...]  # 6 strategies
    
    # Phase 4: Zoom parameters
    zoom_enabled: bool = True
    zoom_radius: int = 2
    
    # General parameters
    max_samples: int = 200
    use_condensed_topics: bool = True
```

## Output

The optimization focuses on tracking high-performing models:

**High-Performance Tracking:**
- `models_90_plus.json` - All models that score 90% or above (with timestamps)
- `top_5_models.json` - Always contains the current top 5 best models

**Progress Tracking:**
- `optimization_live_results.json` - Minimal progress updates

**Key Features:**
- **90%+ models**: Automatically logged when found with immediate notification
- **Top 5 models**: Updated continuously and displayed every 20 configurations
- **Minimal logging**: No extensive logging of all configurations
- **Focus on excellence**: Tracks only the best performing models

## Monitoring Progress

### View All 90%+ Models
```bash
python match-and-choose-model-1/view_90_plus_models.py
```

### View Current Top 5 Models
```bash
python match-and-choose-model-1/view_top_models.py
```

### Monitor Live Progress
```bash
python match-and-choose-model-1/monitor_results.py
```

### Files Generated
- `models_90_plus.json` - All models scoring 90%+ with timestamps
- `top_5_models.json` - Current top 5 models with full details
- `optimization_live_results.json` - Minimal progress tracking

## Key Features

- **Local Model Downloads**: All embedding models are downloaded locally, no online API calls
- **Hierarchical Approach**: Focuses on most promising parameter regions
- **Efficient Search**: Tests ~363 configurations vs exhaustive search of thousands
- **Zoom Capability**: Fine-tunes promising configurations
- **Progress Tracking**: Saves results after each phase for resumability
- **Cloud-Friendly**: Minimal dependencies, works on ucloud instances

## Expected Performance

- **Runtime**: 2-4 hours on ucloud instance
- **Target Accuracy**: 90%+ on topic classification
- **Memory Usage**: ~4GB for largest embedding models
- **Storage**: ~2GB for downloaded models

## Dependencies

```bash
pip install sentence-transformers scikit-learn rank_bm25 tqdm torch
```

## Files

- `optimize_topic_model.py`: Main optimization script
- `config.py`: Configuration management
- `model.py`: Core model implementation
- `search.py`: Search functionality
- `llm.py`: LLM integration
- `evaluate.py`: Evaluation utilities