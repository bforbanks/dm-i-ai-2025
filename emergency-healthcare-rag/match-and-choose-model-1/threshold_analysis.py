#!/usr/bin/env python3
"""
Comprehensive threshold analysis for gap-based decision making
Shows what gets caught vs missed at each threshold
"""

import json
import numpy as np
from pathlib import Path
from search import get_top_topics_with_scores

def analyze_thresholds():
    """Analyze different gap thresholds and their impact"""
    
    # Load train data
    statements_dir = Path("data/train/statements")
    answers_dir = Path("data/train/answers")
    
    # Get all samples for comprehensive analysis
    samples = []
    statement_files = sorted(list(statements_dir.glob("statement_*.txt")))
    
    for statement_file in statement_files:
        statement_num = statement_file.stem.split('_')[1]
        answer_file = answers_dir / f"statement_{statement_num}.json"
        
        if answer_file.exists():
            with open(statement_file, 'r') as f:
                statement = f.read().strip()
            
            with open(answer_file, 'r') as f:
                answer_data = json.load(f)
            
            samples.append({
                'statement': statement,
                'expected_topic': answer_data['statement_topic'],
                'sample_id': int(statement_num)
            })
    
    print("🔍 COMPREHENSIVE THRESHOLD ANALYSIS")
    print("=" * 80)
    print(f"Analyzing {len(samples)} samples...")
    print()
    
    # Analyze each sample
    gap_data = []
    
    for i, sample in enumerate(samples):
        statement = sample['statement']
        expected_topic = sample['expected_topic']
        
        # Get top 5 topics with scores
        top_topics = get_top_topics_with_scores(statement, top_k=5)
        
        # Handle cases with less than 2 topics
        if len(top_topics) < 2:
            # For these cases, we can't calculate a gap, so we'll include them in "above threshold"
            # since they have no competition (only 1 or 0 topics)
            expected_rank = -1
            expected_score = 0.0
            first_score = 0.0
            second_score = 0.0
            gap_1st_to_2nd = 999  # Large gap to ensure they're above any threshold
            
            if len(top_topics) == 1:
                first_score = top_topics[0]['score']
                if top_topics[0]['topic_id'] == expected_topic:
                    expected_rank = 1
                    expected_score = first_score
            elif len(top_topics) == 0:
                # No topics found, mark as incorrect
                expected_rank = -1
                expected_score = 0.0
            
            is_first_correct = expected_rank == 1
            
            gap_data.append({
                'sample_id': sample['sample_id'],
                'expected_rank': expected_rank,
                'expected_score': expected_score,
                'first_score': first_score,
                'second_score': second_score,
                'gap_1st_to_2nd': gap_1st_to_2nd,
                'is_first_correct': is_first_correct
            })
            continue
        
        # Find expected topic rank and score
        expected_rank = -1
        expected_score = 0.0
        
        for rank, topic_info in enumerate(top_topics, 1):
            if topic_info['topic_id'] == expected_topic:
                expected_rank = rank
                expected_score = topic_info['score']
                break
        
        # Calculate gap between 1st and 2nd
        first_score = top_topics[0]['score']
        second_score = top_topics[1]['score'] if len(top_topics) > 1 else 0
        gap_1st_to_2nd = first_score - second_score
        
        # Check if 1st rank is correct
        is_first_correct = expected_rank == 1
        
        gap_data.append({
            'sample_id': sample['sample_id'],
            'expected_rank': expected_rank,
            'expected_score': expected_score,
            'first_score': first_score,
            'second_score': second_score,
            'gap_1st_to_2nd': gap_1st_to_2nd,
            'is_first_correct': is_first_correct
        })
    
    # Define thresholds to test (NA, 0, 1, 2, 3, 5, 10, 15, 20, 30, 40)
    thresholds = ['NA', 0, 1, 2, 3, 5, 10, 15, 20, 30, 40]
    
    print("📊 THRESHOLD ANALYSIS RESULTS")
    print("=" * 80)
    print("Threshold between 1st and 2nd picks: How much better the 1st topic score is than the 2nd topic score")
    print("Above threshold: Cases where gap > threshold (will use topic model 1st pick)")
    print("Below threshold: Cases where gap ≤ threshold (LLM will choose between candidates)")
    print()
    print("Threshold: Threshold value (difference between 1st and 2nd topic scores)")
    print("Total >threshold: Samples where gap > threshold (will use topic model 1st pick)")
    print("✓ >: Above threshold samples where 1st pick was correct")
    print("✗ >: Above threshold samples where 1st pick was wrong")
    print("% ✓: Accuracy percentage for above threshold cases")
    print("Total <threshold: Samples where gap ≤ threshold (LLM will choose between candidates)")
    print("✓ <: Below threshold samples where 1st pick was correct")
    print("✗ <: Below threshold samples where 1st pick was wrong")
    print("% ✓: Accuracy percentage for below threshold cases")
    print()
    print("Threshold | Total >threshold | ✓ > | ✗ > | % ✓ | Total <threshold | ✓ < | ✗ < | % ✓")
    print("          | (no LLM pick)   |     |     |      | (LLM pick)      |     |     |      ")
    print("-" * 100)
    
    for threshold in thresholds:
        if threshold == 'NA':
            # NA means no threshold - all cases use topic model 1st pick
            above_threshold = gap_data
            correct_above = [d for d in above_threshold if d['is_first_correct']]
            incorrect_above = [d for d in above_threshold if not d['is_first_correct']]
            below_threshold = []
            correct_below = []
            incorrect_below = []
        elif threshold == 0:
            # For threshold 0: only go to LLM if gap exactly equals 0
            above_threshold = [d for d in gap_data if d['gap_1st_to_2nd'] > 0]
            correct_above = [d for d in above_threshold if d['is_first_correct']]
            incorrect_above = [d for d in above_threshold if not d['is_first_correct']]
            
            # Cases where gap equals 0 (exact same scores)
            below_threshold = [d for d in gap_data if d['gap_1st_to_2nd'] == 0]
            correct_below = [d for d in below_threshold if d['is_first_correct']]
            incorrect_below = [d for d in below_threshold if not d['is_first_correct']]
        else:
            # Cases above threshold (strictly greater than)
            above_threshold = [d for d in gap_data if d['gap_1st_to_2nd'] > threshold]
            correct_above = [d for d in above_threshold if d['is_first_correct']]
            incorrect_above = [d for d in above_threshold if not d['is_first_correct']]
            
            # Cases below threshold (less than or equal to)
            below_threshold = [d for d in gap_data if d['gap_1st_to_2nd'] <= threshold]
            correct_below = [d for d in below_threshold if d['is_first_correct']]
            incorrect_below = [d for d in below_threshold if not d['is_first_correct']]
        
        # Calculate percentages
        above_accuracy = (len(correct_above) / len(above_threshold) * 100) if above_threshold else 0
        below_accuracy = (len(correct_below) / len(below_threshold) * 100) if below_threshold else 0
        
        print(f"{threshold:>9} | {len(above_threshold):16d} | {len(correct_above):3d} | {len(incorrect_above):3d} | {above_accuracy:5.1f}% | {len(below_threshold):16d} | {len(correct_below):3d} | {len(incorrect_below):3d} | {below_accuracy:5.1f}%")
    
    print()
    
    # Summary recommendations
    print("💡 SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    
    # Find optimal threshold
    best_threshold = None
    best_accuracy = 0
    
    for threshold in thresholds:
        if threshold == 'NA':
            continue  # Skip NA for optimal threshold calculation
        above_threshold = [d for d in gap_data if d['gap_1st_to_2nd'] > threshold]
        if above_threshold:
            correct_above = [d for d in above_threshold if d['is_first_correct']]
            accuracy = len(correct_above) / len(above_threshold)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
    
    print(f"🎯 OPTIMAL THRESHOLD: {best_threshold} points")
    print(f"   Accuracy: {best_accuracy*100:.1f}%")
    print()
    
    # Show what happens at different thresholds
    print("📊 THRESHOLD IMPACT:")
    print("   • Low threshold (5-15): High coverage, lower accuracy")
    print("   • Medium threshold (20-35): Balanced coverage and accuracy")
    print("   • High threshold (40+): High accuracy, lower coverage")
    print()
    
    print("🚀 IMPLEMENTATION STRATEGY:")
    print("   • Use threshold to decide when to trust 1st rank vs use LLM choice")
    print("   • Below threshold: Use LLM to choose between top 2-3 topics")
    print("   • Above threshold: Trust 1st rank and proceed with truth evaluation")
    print("   • Consider adaptive thresholds based on confidence requirements")
    print()
    
    # Print the table again at the end
    print("📊 THRESHOLD ANALYSIS TABLE")
    print("=" * 100)
    print("Threshold between 1st and 2nd picks: How much better the 1st topic score is than the 2nd topic score")
    print("Above threshold: Cases where gap > threshold (will use topic model 1st pick)")
    print("Below threshold: Cases where gap ≤ threshold (LLM will choose between candidates)")
    print()
    print("Threshold: Threshold value (difference between 1st and 2nd topic scores)")
    print("Total >threshold: Samples where gap > threshold (will use topic model 1st pick)")
    print("✓ >: Above threshold samples where 1st pick was correct")
    print("✗ >: Above threshold samples where 1st pick was wrong")
    print("% ✓: Accuracy percentage for above threshold cases")
    print("Total <threshold: Samples where gap ≤ threshold (LLM will choose between candidates)")
    print("✓ <: Below threshold samples where 1st pick was correct")
    print("✗ <: Below threshold samples where 1st pick was wrong")
    print("% ✓: Accuracy percentage for below threshold cases")
    print()
    print("Threshold | Total >threshold | ✓ > | ✗ > | % ✓ | Total <threshold | ✓ < | ✗ < | % ✓")
    print("          | (no LLM pick)   |     |     |      | (LLM pick)      |     |     |      ")
    print("-" * 100)
    
    for threshold in thresholds:
        if threshold == 'NA':
            # NA means no threshold - all cases use topic model 1st pick
            above_threshold = gap_data
            correct_above = [d for d in above_threshold if d['is_first_correct']]
            incorrect_above = [d for d in above_threshold if not d['is_first_correct']]
            below_threshold = []
            correct_below = []
            incorrect_below = []
        elif threshold == 0:
            # For threshold 0: only go to LLM if gap exactly equals 0
            above_threshold = [d for d in gap_data if d['gap_1st_to_2nd'] > 0]
            correct_above = [d for d in above_threshold if d['is_first_correct']]
            incorrect_above = [d for d in above_threshold if not d['is_first_correct']]
            
            # Cases where gap equals 0 (exact same scores)
            below_threshold = [d for d in gap_data if d['gap_1st_to_2nd'] == 0]
            correct_below = [d for d in below_threshold if d['is_first_correct']]
            incorrect_below = [d for d in below_threshold if not d['is_first_correct']]
        else:
            # Cases above threshold (strictly greater than)
            above_threshold = [d for d in gap_data if d['gap_1st_to_2nd'] > threshold]
            correct_above = [d for d in above_threshold if d['is_first_correct']]
            incorrect_above = [d for d in above_threshold if not d['is_first_correct']]
            
            # Cases below threshold (less than or equal to)
            below_threshold = [d for d in gap_data if d['gap_1st_to_2nd'] <= threshold]
            correct_below = [d for d in below_threshold if d['is_first_correct']]
            incorrect_below = [d for d in below_threshold if not d['is_first_correct']]
        
        # Calculate percentages
        above_accuracy = (len(correct_above) / len(above_threshold) * 100) if above_threshold else 0
        below_accuracy = (len(correct_below) / len(below_threshold) * 100) if below_threshold else 0
        
        
        print(f"{threshold:>9} | {len(above_threshold):16d} | {len(correct_above):3d} | {len(incorrect_above):3d} | {above_accuracy:5.1f}% | {len(below_threshold):16d} | {len(correct_below):3d} | {len(incorrect_below):3d} | {below_accuracy:5.1f}%")

if __name__ == "__main__":
    analyze_thresholds() 