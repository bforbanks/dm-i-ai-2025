# 🏎️ Expert System Testing Guide

## Quick Start

You now have multiple tools to test and improve the racing AI:

### 1. 🔧 Test a Single Model
```bash
python test_expert_system.py
```
- Choose from 5 different models
- Options: Quick test, detailed analysis, or visual debugging
- Gets detailed performance metrics and identifies issues

### 2. 🏁 Compare All Models
```bash
python compare_models.py  
```
- Tests all models automatically
- Shows head-to-head comparison
- Identifies the best performer

### 3. 🎮 Watch Your Model Race
```bash
python run.py
```
- Visual gameplay with your selected model
- Currently set to `ImprovedExpertSystem`
- Change `model_name` in run.py to test others

## Available Models

### 🚀 **ImprovedExpertSystem** (Recommended)
- Simplified logic based on baseline analysis
- Conservative but effective collision avoidance
- Target velocity: 14 (safe but fast)
- Clear priority-based decisions

### ⚡ **SimpleExpertSystem** (Ultra-Simple)
- Minimal logic based on proven baselines
- Just front/back sensor logic
- Very robust, hard to break

### 🧠 **OptimalExpertSystem** (Original Complex)
- Advanced sensor fusion and prediction
- May be over-engineered
- Good for learning but potentially unstable

### 📊 **Baselines** (BaselineV, BaselineL)
- Original working models
- Good for comparison

## Testing Strategy

### Phase 1: Quick Comparison
```bash
python compare_models.py
```
This will show you which model performs best overall.

### Phase 2: Detailed Analysis
```bash
python test_expert_system.py
# Choose your best model from Phase 1
# Run detailed test (option 2)
```

### Phase 3: Visual Debugging
```bash
python test_expert_system.py
# Choose visual test (option 3)
# Watch the model play and identify issues
```

### Phase 4: Optimization
Edit your chosen model based on test results, then repeat!

## Reading Test Results

### Key Metrics:
- **Average Distance**: How far the car travels on average
- **Crash Rate**: Percentage of games ending in crashes  
- **Average Velocity**: Speed efficiency
- **Max Distance**: Best single performance

### Good Targets:
- Distance: >2000 (excellent), >1000 (good)
- Crash Rate: <30% (excellent), <50% (acceptable)
- Velocity: >12 (fast), >10 (good)

### Common Issues:
- **High Crash Rate**: Collision avoidance too weak
- **Low Velocity**: Too conservative, need more acceleration
- **Low Distance**: Either crashes too much or too slow

## Improving Performance

### If crashes are high:
1. Increase safety thresholds (danger_threshold, caution_threshold)
2. Make lane changes more conservative
3. Add more braking in emergency situations

### If velocity is low:
1. Lower target_velocity or increase max_velocity
2. Accelerate more aggressively when safe
3. Reduce unnecessary braking

### If stuck/not progressing:
1. Add stuck detection and recovery
2. Ensure the model can accelerate from standstill
3. Check for logic loops

## Pro Tips

1. **Start Simple**: Use SimpleExpertSystem as baseline
2. **Test Frequently**: Small changes, frequent testing
3. **Watch Visual**: Visual testing reveals logic issues
4. **Compare**: Always compare against baselines
5. **Iterate**: Make one change at a time

## File Structure
```
race-car/
├── models/
│   ├── OptimalExpertSystem.py     # Original complex system
│   ├── ImprovedExpertSystem.py    # Simplified + ultra-simple
│   ├── BaselineV.py               # Baseline velocity model
│   └── BaselineL.py               # Baseline lane model
├── test_expert_system.py          # Single model testing
├── compare_models.py              # Multi-model comparison  
├── run.py                         # Visual gameplay
└── TESTING_GUIDE.md              # This guide
```

Happy racing! 🏁