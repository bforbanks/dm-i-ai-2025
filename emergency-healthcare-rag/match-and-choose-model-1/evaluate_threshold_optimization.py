#!/usr/bin/env python3
"""
Advanced threshold optimization for match-and-choose model
Focuses on optimizing when to use LLM vs topic model decisions
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Local imports
try:
    from .search import get_top_topics_with_scores, load_topics_mapping
    from .llm import classify_topic_and_truth, classify_truth_only
except ImportError:
    from search import get_top_topics_with_scores, load_topics_mapping  
    from llm import classify_topic_and_truth, classify_truth_only

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------

@dataclass 
class ThresholdConfig:
    """Configuration for threshold optimization"""
    # Threshold range to test
    threshold_min: float = 0.0
    threshold_max: float = 20.0
    threshold_step: float = 0.5
    
    # Special thresholds to test
    special_thresholds: List[str] = None  # Will default to ['NA']
    
    # Evaluation parameters
    max_samples: int = 200
    top_k_candidates: int = 5
    
    # Analysis parameters
    analyze_failure_cases: bool = True
    save_detailed_results: bool = True
    create_visualizations: bool = True
    
    def __post_init__(self):
        if self.special_thresholds is None:
            self.special_thresholds = ['NA']

# Global paths
STATEMENT_DIR = Path("data/train/statements")
ANSWER_DIR = Path("data/train/answers")

# -----------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------

def load_statements() -> List[Tuple[str, int]]:
    """Load training statements with true topics"""
    records = []
    for path in sorted(STATEMENT_DIR.glob("*.txt")):
        sid = path.stem.split("_")[1]
        statement = path.read_text().strip()
        ans = json.loads((ANSWER_DIR / f"statement_{sid}.json").read_text())
        records.append((statement, ans["statement_topic"]))
    return records

# -----------------------------------------------------------------------
# ANALYSIS FUNCTIONS  
# -----------------------------------------------------------------------

def analyze_search_performance(statements: List[Tuple[str, int]], top_k: int = 10) -> Dict:
    """Analyze baseline search performance without LLM intervention"""
    print("🔍 Analyzing baseline search performance...")
    
    results = []
    score_gaps = []
    rank_distributions = {}
    
    for statement, true_topic in tqdm(statements, desc="Baseline analysis"):
        top_topics = get_top_topics_with_scores(statement, top_k=top_k)
        
        # Find rank of correct topic
        rank_correct = 0
        for rank, topic_result in enumerate(top_topics, 1):
            if topic_result['topic_id'] == true_topic:
                rank_correct = rank
                break
        
        # Calculate score gap between 1st and 2nd
        gap = 0.0
        if len(top_topics) >= 2:
            gap = top_topics[0]['score'] - top_topics[1]['score']
        
        results.append({
            'statement': statement,
            'true_topic': true_topic,
            'rank_correct': rank_correct,
            'gap': gap,
            'top_topics': top_topics[:3]
        })
        
        score_gaps.append(gap)
        rank_distributions[rank_correct] = rank_distributions.get(rank_correct, 0) + 1
    
    # Calculate metrics
    total = len(results)
    top1_accuracy = sum(1 for r in results if r['rank_correct'] == 1) / total
    top3_accuracy = sum(1 for r in results if r['rank_correct'] <= 3) / total
    top5_accuracy = sum(1 for r in results if r['rank_correct'] <= 5) / total
    
    return {
        'results': results,
        'metrics': {
            'total_samples': total,
            'top1_accuracy': top1_accuracy,
            'top3_accuracy': top3_accuracy,
            'top5_accuracy': top5_accuracy,
            'rank_distribution': rank_distributions,
            'score_gaps': {
                'mean': np.mean(score_gaps),
                'std': np.std(score_gaps),
                'percentiles': {
                    'p10': np.percentile(score_gaps, 10),
                    'p25': np.percentile(score_gaps, 25),
                    'p50': np.percentile(score_gaps, 50),
                    'p75': np.percentile(score_gaps, 75),
                    'p90': np.percentile(score_gaps, 90)
                }
            }
        }
    }

def simulate_threshold_strategy(baseline_results: List[Dict], threshold: float, 
                              llm_topic_accuracy: float = 0.75, 
                              llm_truth_accuracy: float = 0.92) -> Dict:
    """
    Simulate performance of threshold-based strategy
    
    Args:
        baseline_results: Results from baseline search analysis
        threshold: Score gap threshold for LLM intervention
        llm_topic_accuracy: Assumed LLM accuracy when choosing between topics
        llm_truth_accuracy: Assumed LLM accuracy for truth classification
    """
    total_samples = len(baseline_results)
    separated_cases = 0
    combined_cases = 0
    correct_topic_predictions = 0
    
    for result in baseline_results:
        gap = result['gap']
        rank_correct = result['rank_correct']
        
        if threshold == 'NA' or (threshold != 'NA' and gap > threshold):
            # Use separated approach (topic model + truth LLM)
            separated_cases += 1
            if rank_correct == 1:  # Topic model got it right
                correct_topic_predictions += 1
        else:
            # Use combined approach (LLM chooses topic + truth)
            combined_cases += 1
            
            # Check if correct topic is in top candidates within threshold
            correct_in_candidates = rank_correct <= 5  # Assume we feed top-5 to LLM
            
            if correct_in_candidates:
                # LLM has a chance to pick correctly
                if np.random.random() < llm_topic_accuracy:
                    correct_topic_predictions += 1
    
    topic_accuracy = correct_topic_predictions / total_samples
    
    return {
        'threshold': threshold,
        'topic_accuracy': topic_accuracy,
        'separated_cases': separated_cases,
        'combined_cases': combined_cases,
        'separated_ratio': separated_cases / total_samples,
        'combined_ratio': combined_cases / total_samples
    }

def find_optimal_thresholds(baseline_analysis: Dict, config: ThresholdConfig) -> Dict:
    """Find optimal thresholds for different objectives"""
    print("🎯 Finding optimal thresholds...")
    
    baseline_results = baseline_analysis['results']
    thresholds_to_test = []
    
    # Add numeric thresholds
    current_threshold = config.threshold_min
    while current_threshold <= config.threshold_max:
        thresholds_to_test.append(current_threshold)
        current_threshold += config.threshold_step
    
    # Add special thresholds
    thresholds_to_test.extend(config.special_thresholds)
    
    threshold_results = []
    
    for threshold in tqdm(thresholds_to_test, desc="Testing thresholds"):
        result = simulate_threshold_strategy(baseline_results, threshold)
        threshold_results.append(result)
    
    # Find optimal thresholds for different objectives
    numeric_results = [r for r in threshold_results if isinstance(r['threshold'], (int, float))]
    
    best_accuracy = max(threshold_results, key=lambda x: x['topic_accuracy'])
    best_balanced = min(numeric_results, key=lambda x: abs(x['separated_ratio'] - 0.5)) if numeric_results else None
    
    # Find threshold that maximizes accuracy improvement over baseline
    baseline_accuracy = baseline_analysis['metrics']['top1_accuracy']
    improvements = [(r['topic_accuracy'] - baseline_accuracy, r) for r in threshold_results]
    best_improvement = max(improvements, key=lambda x: x[0])[1]
    
    return {
        'all_results': threshold_results,
        'baseline_accuracy': baseline_accuracy,
        'optimal_thresholds': {
            'best_accuracy': best_accuracy,
            'best_improvement': best_improvement,
            'best_balanced': best_balanced
        }
    }

def analyze_complementary_models(baseline_analysis: Dict) -> Dict:
    """Analyze where the current model fails and suggest complementary approaches"""
    print("🔍 Analyzing failure cases for complementary model insights...")
    
    results = baseline_analysis['results']
    
    # Categorize failure cases
    failure_categories = {
        'close_scores': [],      # Cases where top scores are very close
        'correct_in_top3': [],   # Cases where correct answer is in top 3 but not 1st
        'correct_in_top5': [],   # Cases where correct answer is in top 5 but not top 3
        'not_found': []          # Cases where correct answer is not in top 5
    }
    
    for result in results:
        rank_correct = result['rank_correct']
        gap = result['gap']
        
        if rank_correct != 1:  # Failure case
            if rank_correct <= 3:
                if gap < 2.0:  # Threshold for "close scores"
                    failure_categories['close_scores'].append(result)
                else:
                    failure_categories['correct_in_top3'].append(result)
            elif rank_correct <= 5:
                failure_categories['correct_in_top5'].append(result)
            else:
                failure_categories['not_found'].append(result)
    
    # Analyze patterns in each category
    category_analysis = {}
    for category, cases in failure_categories.items():
        if not cases:
            continue
            
        avg_gap = np.mean([case['gap'] for case in cases])
        
        # Analyze statement characteristics
        statement_lengths = [len(case['statement'].split()) for case in cases]
        avg_length = np.mean(statement_lengths)
        
        category_analysis[category] = {
            'count': len(cases),
            'percentage': len(cases) / len(results) * 100,
            'avg_gap': avg_gap,
            'avg_statement_length': avg_length,
            'examples': cases[:3]  # Store first 3 examples
        }
    
    # Calculate potential for improvement
    total_failures = len([r for r in results if r['rank_correct'] != 1])
    recoverable_failures = len(failure_categories['close_scores']) + len(failure_categories['correct_in_top3'])
    recovery_potential = recoverable_failures / len(results) if results else 0
    
    return {
        'failure_categories': category_analysis,
        'total_failures': total_failures,
        'recoverable_failures': recoverable_failures,
        'recovery_potential': recovery_potential,
        'recommendations': generate_complementary_recommendations(category_analysis)
    }

def generate_complementary_recommendations(category_analysis: Dict) -> List[str]:
    """Generate recommendations for complementary models based on failure analysis"""
    recommendations = []
    
    if 'close_scores' in category_analysis and category_analysis['close_scores']['count'] > 10:
        recommendations.append(
            "🔄 High potential for LLM intervention: Many cases have close scores between top candidates. "
            f"LLM could improve {category_analysis['close_scores']['percentage']:.1f}% of cases."
        )
    
    if 'correct_in_top3' in category_analysis and category_analysis['correct_in_top3']['count'] > 5:
        recommendations.append(
            "🎯 Semantic search could help: Correct topics often appear in top-3 but miss 1st place. "
            "A semantic model might capture different relevance signals."
        )
    
    if 'not_found' in category_analysis and category_analysis['not_found']['count'] > 5:
        recommendations.append(
            "📚 Consider expanding search scope: Some correct topics don't appear in top-5. "
            "Different chunking strategy or broader semantic search might help."
        )
    
    # Analyze statement lengths for targeted recommendations
    long_statements = [cat for cat in category_analysis.values() if cat.get('avg_statement_length', 0) > 20]
    if long_statements:
        recommendations.append(
            "📝 Long statements need different handling: Consider query summarization or "
            "multiple search strategies for complex medical statements."
        )
    
    return recommendations

def create_visualizations(baseline_analysis: Dict, threshold_analysis: Dict, 
                         complementary_analysis: Dict, config: ThresholdConfig):
    """Create visualization plots for the analysis"""
    if not config.create_visualizations:
        return
    
    print("📊 Creating visualizations...")
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Topic Model Optimization Analysis', fontsize=16)
        
        # 1. Score gap distribution
        gaps = [r['gap'] for r in baseline_analysis['results']]
        axes[0, 0].hist(gaps, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Score Gap (1st - 2nd)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Score Gaps')
        axes[0, 0].axvline(np.mean(gaps), color='red', linestyle='--', label=f'Mean: {np.mean(gaps):.2f}')
        axes[0, 0].legend()
        
        # 2. Threshold performance
        threshold_results = threshold_analysis['all_results']
        numeric_results = [r for r in threshold_results if isinstance(r['threshold'], (int, float))]
        
        if numeric_results:
            thresholds = [r['threshold'] for r in numeric_results]
            accuracies = [r['topic_accuracy'] for r in numeric_results]
            
            axes[0, 1].plot(thresholds, accuracies, 'b-', marker='o', markersize=4)
            axes[0, 1].set_xlabel('Threshold')
            axes[0, 1].set_ylabel('Topic Accuracy')
            axes[0, 1].set_title('Accuracy vs Threshold')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Mark optimal threshold
            best_threshold = threshold_analysis['optimal_thresholds']['best_accuracy']
            axes[0, 1].axvline(best_threshold['threshold'], color='red', linestyle='--', 
                              label=f'Best: {best_threshold["threshold"]} ({best_threshold["topic_accuracy"]:.3f})')
            axes[0, 1].legend()
        
        # 3. Rank distribution
        rank_dist = baseline_analysis['metrics']['rank_distribution']
        ranks = list(rank_dist.keys())
        counts = list(rank_dist.values())
        
        axes[1, 0].bar(ranks, counts, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Rank of Correct Topic')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Distribution of Correct Topic Ranks')
        axes[1, 0].set_xticks(ranks)
        
        # 4. Failure category breakdown
        failure_cats = complementary_analysis['failure_categories']
        cat_names = list(failure_cats.keys())
        cat_counts = [failure_cats[cat]['count'] for cat in cat_names]
        
        axes[1, 1].pie(cat_counts, labels=cat_names, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Failure Case Categories')
        
        plt.tight_layout()
        plt.savefig('threshold_optimization_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Saved visualizations to: threshold_optimization_analysis.png")
        
    except ImportError:
        print("⚠️ matplotlib not available, skipping visualizations")
    except Exception as e:
        print(f"⚠️ Error creating visualizations: {e}")

# -----------------------------------------------------------------------
# MAIN EVALUATION
# -----------------------------------------------------------------------

def run_threshold_optimization():
    """Run comprehensive threshold optimization analysis"""
    print("🎯 Starting Threshold Optimization Analysis")
    print("=" * 60)
    
    config = ThresholdConfig()
    
    # Load test data
    print("📚 Loading evaluation data...")
    statements = load_statements()
    if config.max_samples and config.max_samples < len(statements):
        statements = statements[:config.max_samples]
    print(f"📊 Evaluating on {len(statements)} statements")
    
    # 1. Baseline analysis
    baseline_analysis = analyze_search_performance(statements, top_k=10)
    
    print(f"\n📈 BASELINE PERFORMANCE:")
    metrics = baseline_analysis['metrics']
    print(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.3f}")
    print(f"  Top-3 Accuracy: {metrics['top3_accuracy']:.3f}")
    print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.3f}")
    print(f"  Mean Score Gap: {metrics['score_gaps']['mean']:.2f} ± {metrics['score_gaps']['std']:.2f}")
    
    # 2. Threshold optimization
    threshold_analysis = find_optimal_thresholds(baseline_analysis, config)
    
    print(f"\n🎯 OPTIMAL THRESHOLDS:")
    optimal = threshold_analysis['optimal_thresholds']
    
    print(f"  Best Accuracy: threshold={optimal['best_accuracy']['threshold']}, "
          f"accuracy={optimal['best_accuracy']['topic_accuracy']:.3f}")
    
    if optimal['best_balanced']:
        print(f"  Most Balanced: threshold={optimal['best_balanced']['threshold']:.1f}, "
              f"separated={optimal['best_balanced']['separated_ratio']:.1%}")
    
    improvement = optimal['best_improvement']['topic_accuracy'] - threshold_analysis['baseline_accuracy']
    print(f"  Best Improvement: +{improvement:.3f} accuracy gain")
    
    # 3. Complementary model analysis
    complementary_analysis = analyze_complementary_models(baseline_analysis)
    
    print(f"\n🔍 FAILURE ANALYSIS:")
    failure_cats = complementary_analysis['failure_categories']
    for category, data in failure_cats.items():
        print(f"  {category.replace('_', ' ').title()}: {data['count']} cases ({data['percentage']:.1f}%)")
    
    print(f"\n💡 RECOVERY POTENTIAL: {complementary_analysis['recovery_potential']:.1%} of cases could be improved")
    
    print(f"\n🚀 RECOMMENDATIONS:")
    for rec in complementary_analysis['recommendations']:
        print(f"  {rec}")
    
    # 4. Create visualizations
    create_visualizations(baseline_analysis, threshold_analysis, complementary_analysis, config)
    
    # 5. Save detailed results
    if config.save_detailed_results:
        results = {
            'config': {
                'threshold_range': (config.threshold_min, config.threshold_max, config.threshold_step),
                'max_samples': config.max_samples,
                'evaluation_timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S')
            },
            'baseline_analysis': baseline_analysis,
            'threshold_analysis': threshold_analysis,
            'complementary_analysis': complementary_analysis
        }
        
        output_file = "threshold_optimization_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
    
    # 6. Implementation guidance
    print(f"\n🛠️ IMPLEMENTATION GUIDANCE:")
    print(f"  1. Set threshold to {optimal['best_accuracy']['threshold']} for maximum accuracy")
    print(f"  2. Expected improvement: {improvement:.1%} over current performance")
    print(f"  3. {optimal['best_accuracy']['separated_ratio']:.1%} of cases will use fast topic model")
    print(f"  4. {optimal['best_accuracy']['combined_ratio']:.1%} of cases will use slower LLM decision")
    
    if complementary_analysis['recovery_potential'] > 0.1:
        print(f"  5. Consider hybrid semantic+BM25 search for additional {complementary_analysis['recovery_potential']:.1%} improvement")
    
    print(f"\n✅ Threshold optimization complete!")

if __name__ == "__main__":
    run_threshold_optimization()