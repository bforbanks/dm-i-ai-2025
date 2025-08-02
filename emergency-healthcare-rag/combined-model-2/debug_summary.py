#!/usr/bin/env python3
"""
Summary of all debugging tools available for combined-model-2
"""

import sys
from pathlib import Path

def print_debug_tools():
    """Print all available debugging tools"""
    
    print("🔧 Combined Model 2 - Debugging Tools")
    print("=" * 60)
    print()
    
    tools = [
        {
            "name": "evaluate.py",
            "description": "Main evaluation script - measures accuracy and timing",
            "usage": "python combined-model-2/evaluate.py",
            "output": "Accuracy metrics, timing, and detailed results saved to JSON"
        },
        {
            "name": "debug_llm_response.py", 
            "description": "Analyzes LLM response parsing and tests different formats",
            "usage": "python combined-model-2/debug_llm_response.py",
            "output": "Raw LLM responses, parsing success/failure, response format analysis"
        },
        {
            "name": "debug_search.py",
            "description": "Analyzes search results and context quality",
            "usage": "python combined-model-2/debug_search.py", 
            "output": "Search ranking, expected topic detection, context relevance analysis"
        },
        {
            "name": "analyze_failures.py",
            "description": "Deep analysis of specific failure cases",
            "usage": "python combined-model-2/analyze_failures.py",
            "output": "Detailed failure analysis, LLM decision reasoning, context quality"
        },
        {
            "name": "timing_analysis.py",
            "description": "Measures time spent in each component of the pipeline",
            "usage": "python combined-model-2/timing_analysis.py",
            "output": "Breakdown of search time, context time, LLM time"
        },
        {
            "name": "model_switcher.py",
            "description": "Lists and switches between different LLM models",
            "usage": "python combined-model-2/model_switcher.py [model_name]",
            "output": "Available models, current model, model switching"
        }
    ]
    
    for i, tool in enumerate(tools, 1):
        print(f"{i}. {tool['name']}")
        print(f"   📝 {tool['description']}")
        print(f"   💻 Usage: {tool['usage']}")
        print(f"   📊 Output: {tool['output']}")
        print()

def print_current_status():
    """Print current model status and configuration"""
    
    print("📊 Current Status")
    print("=" * 60)
    
    try:
        from config import get_llm_model, get_model_info
        model = get_llm_model()
        model_info = get_model_info(model)
        
        print(f"🤖 Current LLM Model: {model_info['name']}")
        print(f"📋 Description: {model_info['description']}")
        print(f"🔢 Parameters: {model_info['parameters']}")
        print(f"🧠 Reasoning: {model_info['reasoning']}")
        print()
        
    except Exception as e:
        print(f"❌ Could not get model info: {e}")
        print()

def print_quick_tests():
    """Print quick test commands"""
    
    print("⚡ Quick Tests")
    print("=" * 60)
    print()
    
    tests = [
        {
            "name": "Basic Evaluation (5 samples)",
            "command": "python combined-model-2/evaluate.py",
            "description": "Test accuracy and timing on 5 samples"
        },
        {
            "name": "Search Debug",
            "command": "python combined-model-2/debug_search.py",
            "description": "Check if search is finding correct topics"
        },
        {
            "name": "LLM Response Debug", 
            "command": "python combined-model-2/debug_llm_response.py",
            "description": "Test LLM response parsing"
        },
        {
            "name": "Model Switching",
            "command": "python combined-model-2/model_switcher.py",
            "description": "List available models and switch between them"
        }
    ]
    
    for test in tests:
        print(f"🔍 {test['name']}")
        print(f"   💻 {test['command']}")
        print(f"   📝 {test['description']}")
        print()

def print_troubleshooting():
    """Print common issues and solutions"""
    
    print("🔧 Troubleshooting Guide")
    print("=" * 60)
    print()
    
    issues = [
        {
            "issue": "LLM model not found",
            "symptoms": "Error calling LLM: model 'xxx' not found",
            "solution": "Use model_switcher.py to switch to an available model, or pull the model with ollama pull"
        },
        {
            "issue": "Import errors",
            "symptoms": "ImportError: attempted relative import with no known parent package",
            "solution": "Run scripts from the emergency-healthcare-rag directory, not from combined-model-2"
        },
        {
            "issue": "Low accuracy",
            "symptoms": "Truth accuracy < 70% or topic accuracy < 80%",
            "solution": "Run debug_search.py to check if search is working, then analyze_failures.py for specific cases"
        },
        {
            "issue": "Slow performance",
            "symptoms": "Average time > 10 seconds per sample",
            "solution": "Run timing_analysis.py to identify bottlenecks, consider switching to smaller model"
        },
        {
            "issue": "LLM parsing errors",
            "symptoms": "invalid literal for int() with base 10",
            "solution": "Run debug_llm_response.py to see raw responses and improve parsing"
        }
    ]
    
    for issue in issues:
        print(f"❌ {issue['issue']}")
        print(f"   🔍 Symptoms: {issue['symptoms']}")
        print(f"   ✅ Solution: {issue['solution']}")
        print()

def main():
    """Main function"""
    print_debug_tools()
    print_current_status()
    print_quick_tests()
    print_troubleshooting()
    
    print("✅ Debug summary complete!")
    print("\n💡 Tip: Start with 'python combined-model-2/evaluate.py' for basic testing")

if __name__ == "__main__":
    main() 