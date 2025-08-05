# Topic Model Optimization Plan

## Overview

This optimization plan implements a comprehensive evaluation strategy for improving the match-and-choose topic model using hybrid BM25 + semantic search approaches and advanced threshold optimization.

## Current Performance (Baseline)
- **BM25-only search**: 89.5% top-1 accuracy
- **Overall system**: 85% topic accuracy, 92% truth accuracy
- **Current approach**: Threshold-based decision making (BM25 + LLM intervention)

## Optimization Objectives

### Primary Goals
1. **Maximize 1st pick accuracy** (current priority)
2. **Minimize gap when correct pick is not 1st** (improve MRR)
3. **Maximize separation when 1st pick is correct** (confidence)
4. **Optimize threshold for LLM intervention** (efficiency vs accuracy trade-off)

### Secondary Goals
- Find complementary models that excel where current model fails
- Identify optimal fusion strategies for BM25 + semantic search
- Provide data-driven threshold recommendations
- Enable smart LLM intervention only when topic model is uncertain

## Evaluation Scripts

### 1. `optimize_topic_model.py` - Hybrid Search Optimization

**Purpose**: Comprehensive evaluation of BM25 + semantic search combinations

**Models Tested**:
- `sentence-transformers/all-MiniLM-L6-v2` (lightweight, 384d)
- `sentence-transformers/all-mpnet-base-v2` (best general performance, 768d)  
- `sentence-transformers/all-distilroberta-v1` (different architecture, 768d)

**Fusion Strategies**:
- `bm25_only` - Current baseline
- `semantic_only` - Pure semantic search baseline
- `linear_0.3/0.5/0.7` - Linear combinations with different weights
- `rrf` - Reciprocal Rank Fusion (proven effective in search)
- `adaptive` - Query-dependent weighting based on characteristics

**Key Metrics**:
- Top-1/3/5 accuracy
- Mean Reciprocal Rank (MRR)
- Score separation when correct (confidence metric)
- Score gap percentiles (threshold optimization data)
- Query processing time

**Expected Outcomes**:
- Identify best complementary model pairs
- Find optimal fusion strategy
- Quantify improvement potential (+2-5% accuracy expected)
- Provide implementation-ready configurations

### 2. `evaluate_threshold_optimization.py` - Threshold Strategy Analysis

**Purpose**: Optimize when to use LLM vs topic model decisions

**Analysis Components**:

#### A. Baseline Search Analysis
- Analyze current BM25 performance without LLM
- Identify failure patterns and characteristics
- Calculate score gap distributions for threshold guidance

#### B. Threshold Strategy Simulation
- Test threshold range 0.0 to 20.0 (step 0.5)
- Simulate LLM intervention based on score gaps
- Model realistic LLM accuracy rates (75% topic choice, 92% truth)
- Find optimal thresholds for different objectives

#### C. Failure Case Analysis
- **Close Scores**: Cases where top candidates have similar scores
- **Correct in Top-3**: Cases where correct answer is 2nd or 3rd
- **Correct in Top-5**: Cases where correct answer is 4th or 5th  
- **Not Found**: Cases where correct answer is not in top-5

#### D. Complementary Model Insights
- Identify patterns in failure cases
- Generate recommendations for hybrid approaches
- Quantify recovery potential for each failure category

**Expected Outcomes**:
- Optimal threshold values for different scenarios
- Clear guidance on when LLM intervention helps
- Failure pattern analysis for targeted improvements
- Recovery potential estimates (+5-10% improvement possible)

## Evaluation Metrics Deep Dive

### 1. Top-K Accuracy
- **Top-1**: Primary metric (current focus)
- **Top-3**: Important for LLM intervention strategy
- **Top-5**: Upper bound for practical LLM candidate sets

### 2. Mean Reciprocal Rank (MRR)
- Measures how close correct answers are to 1st position
- Critical for understanding improvement potential
- Formula: `MRR = 1/N * Σ(1/rank_i)` where rank_i is position of correct answer

### 3. Score Separation Analysis
- **When 1st pick is correct**: Gap between 1st and 2nd scores
- **Higher separation = higher confidence** in correct predictions
- Used to optimize threshold values for LLM intervention

### 4. Confidence Metrics
- **Score Gap Percentiles**: P25, P50, P75, P90 for threshold selection
- **Gap Distribution**: Understanding when model is uncertain
- **Threshold Sensitivity**: How performance changes with different gaps

## Implementation Strategy

### Phase 1: Baseline Analysis (Current Status)
- [x] BM25-only implementation with optimized chunking
- [x] Threshold-based LLM intervention (current system)
- [x] Basic evaluation on train/validation sets

### Phase 2: Hybrid Search Optimization
```bash
# Run comprehensive hybrid search evaluation
cd emergency-healthcare-rag/
python match-and-choose-model-1/optimize_topic_model.py
```

**Expected Runtime**: 2-4 hours (depending on hardware)
**Output**: `optimization_results_topic_model.json`

### Phase 3: Threshold Optimization
```bash 
# Run threshold analysis and failure case study
python match-and-choose-model-1/evaluate_threshold_optimization.py
```

**Expected Runtime**: 30-60 minutes
**Output**: `threshold_optimization_results.json` + visualizations

### Phase 4: Integration and Testing
1. Implement best hybrid search configuration
2. Update threshold values based on analysis
3. A/B test against current system
4. Validate on held-out test set

## Expected Improvements

### Conservative Estimates
- **Hybrid Search**: +2-3% top-1 accuracy (89.5% → 92-93%)
- **Threshold Optimization**: +1-2% from better LLM intervention
- **Combined**: +3-5% total improvement (89.5% → 92.5-94.5%)

### Aggressive Estimates (if complementary models work well)
- **Hybrid Search**: +4-6% top-1 accuracy (89.5% → 93.5-95.5%)
- **Threshold Optimization**: +2-3% from smarter LLM use
- **Combined**: +6-9% total improvement (89.5% → 95.5-98.5%)

### Efficiency Gains
- **Threshold Optimization**: Only use slow LLM when needed (20-40% of cases)
- **Caching**: Precomputed embeddings for faster semantic search
- **Model Selection**: Choose optimal speed/accuracy trade-off

## Success Criteria

### Must-Have (Required for Success)
- [x] Scripts execute without errors on ucloud
- [ ] Top-1 accuracy improvement >= 2%
- [ ] Comprehensive evaluation on 200 train samples
- [ ] Clear implementation recommendations
- [ ] Threshold values with theoretical justification

### Nice-to-Have (Bonus Objectives)  
- [ ] Hybrid model outperforms both BM25 and semantic individually
- [ ] Identify failure patterns that complement each other
- [ ] Visualizations showing optimization landscape
- [ ] Recovery potential analysis for future work

## Files and Dependencies

### New Files Created
- `optimize_topic_model.py` - Main hybrid search evaluation
- `evaluate_threshold_optimization.py` - Threshold analysis
- `OPTIMIZATION_PLAN.md` - This planning document

### Dependencies Required
```bash
pip install sentence-transformers scikit-learn matplotlib numpy tqdm
```

### Existing Files Used
- `search.py` - Current BM25 implementation
- `llm.py` - LLM classification functions
- `model.py` - Current threshold-based logic
- `config.py` - Configuration management

## Usage Instructions

### 1. Preparation
```bash
# Ensure you're in the emergency-healthcare-rag directory
cd emergency-healthcare-rag/

# Install additional dependencies (if not already installed)
pip install sentence-transformers scikit-learn matplotlib
```

### 2. Run Hybrid Search Optimization
```bash
# Full evaluation (recommended)
python match-and-choose-model-1/optimize_topic_model.py

# Monitor progress - should take 2-4 hours
# Output: optimization_results_topic_model.json
```

### 3. Run Threshold Analysis  
```bash
# Threshold optimization and failure analysis
python match-and-choose-model-1/evaluate_threshold_optimization.py

# Faster execution - should take 30-60 minutes
# Output: threshold_optimization_results.json + visualizations
```

### 4. Review Results
```bash
# Check main results
cat optimization_results_topic_model.json | jq '.best_configurations[:5]'

# Check threshold recommendations  
cat threshold_optimization_results.json | jq '.threshold_analysis.optimal_thresholds'
```

## Next Steps After Evaluation

### 1. Implementation
- Update `search.py` with best hybrid configuration
- Modify `config.py` with optimal threshold values
- Test integration with existing LLM pipeline

### 2. Validation
- Run full evaluation on validation set
- Compare against current system performance
- Verify improvements are statistically significant

### 3. Production Deployment
- Update model pipeline with optimized configuration
- Monitor performance on real workloads
- Collect feedback for further iterations

## Risk Mitigation

### Potential Issues
- **Model Download Failures**: Scripts include error handling for model loading
- **Memory Constraints**: Embeddings are cached and processed in batches
- **Runtime Performance**: Focus on practical configurations for production use
- **Overfitting**: Evaluation uses train set, but provides guidance for validation

### Fallback Plans
- If hybrid search doesn't improve: Fall back to threshold optimization only
- If semantic models fail to load: Continue with BM25-only threshold analysis
- If evaluation takes too long: Reduce sample size with `max_samples` parameter

## Success Metrics Summary

| Metric | Current | Target | Stretch Goal |
|--------|---------|--------|-------------|
| Top-1 Accuracy | 89.5% | 92-93% | 95%+ |
| Top-3 Accuracy | 97.0% | 98%+ | 99%+ |
| MRR | ~0.93 | 0.95+ | 0.97+ |
| LLM Usage | Variable | 20-40% | Optimized |
| Query Time | ~0.09s | <0.15s | <0.10s |

---

This optimization plan provides a systematic approach to improving the topic model through both hybrid search strategies and threshold optimization, with clear metrics and expected outcomes for evaluation on ucloud.