#!/usr/bin/env python3
"""
Quick start script for the hybrid retrieval pipeline.
Runs testing, optimization, and analysis in sequence.
"""

import subprocess
import sys
from pathlib import Path
import time

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {cmd}")
    print('='*60)
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        print(f"✓ {description} completed successfully in {elapsed:.1f} seconds")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"✗ {description} failed after {elapsed:.1f} seconds")
        print(f"Error: {e}")
        if e.stdout:
            print("Stdout:")
            print(e.stdout)
        if e.stderr:
            print("Stderr:")
            print(e.stderr)
        return False

def main():
    """Run the complete pipeline."""
    print("🚀 HYBRID RETRIEVAL PIPELINE - QUICK START")
    print("="*60)
    
    # Check if we're in the right directory
    if not Path("hybrid_retrieval_bo.py").exists():
        print("Error: Please run this script from the ejRAGv2 directory")
        sys.exit(1)
    
    # Step 1: Install dependencies
    print("\n📦 Step 1: Installing dependencies...")
    if not run_command("pip install -r requirements.txt", "Dependency installation"):
        print("Failed to install dependencies. Please install manually:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Step 2: Run tests
    print("\n🧪 Step 2: Running tests...")
    if not run_command("python test_hybrid.py", "Test suite"):
        print("Tests failed! Please fix issues before continuing.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Step 3: Run optimization
    print("\n🔍 Step 3: Running Bayesian optimization...")
    print("This may take a while depending on your hardware.")
    
    # Ask user for optimization parameters
    print("\nOptimization parameters:")
    init_points = input("Number of initial random points (default: 10): ").strip()
    init_points = int(init_points) if init_points else 10
    
    n_iter = input("Number of optimization iterations (default: 40): ").strip()
    n_iter = int(n_iter) if n_iter else 40
    
    print(f"\nRunning optimization with {init_points} init points and {n_iter} iterations...")
    
    # Modify the script to use custom parameters
    with open("hybrid_retrieval_bo.py", "r") as f:
        content = f.read()
    
    # Replace the default parameters
    content = content.replace(
        "optimizer = run_bayesian_optimization(init_points=10, n_iter=40)",
        f"optimizer = run_bayesian_optimization(init_points={init_points}, n_iter={n_iter})"
    )
    
    with open("hybrid_retrieval_bo.py", "w") as f:
        f.write(content)
    
    if not run_command("python hybrid_retrieval_bo.py", "Bayesian optimization"):
        print("Optimization failed!")
        sys.exit(1)
    
    # Step 4: Analyze results
    print("\n📊 Step 4: Analyzing results...")
    if not run_command("python analyze_results.py", "Results analysis"):
        print("Analysis failed, but optimization results are still available.")
    
    # Step 5: Summary
    print("\n🎉 PIPELINE COMPLETE!")
    print("="*60)
    print("Generated files:")
    
    results_files = [
        "hybrid_bo_results.json",
        "analysis_output/optimization_progress.png",
        "analysis_output/parameter_importance.png", 
        "analysis_output/parameter_distributions.png",
        "analysis_output/optimization_summary.txt"
    ]
    
    for file_path in results_files:
        if Path(file_path).exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (not found)")
    
    print("\nNext steps:")
    print("1. Check hybrid_bo_results.json for the best configuration")
    print("2. View analysis_output/ for visualizations and reports")
    print("3. Use the best parameters in your production system")
    
    # Show best results if available
    try:
        import json
        with open("hybrid_bo_results.json", "r") as f:
            results = json.load(f)
        
        best = results['best_config']
        print(f"\n🏆 BEST CONFIGURATION FOUND:")
        print(f"   Top-1 Accuracy: {best['target']:.4f}")
        print(f"   Model: {'MiniLM' if best['params']['model_selector'] < 0.5 else 'ColBERT'}")
        print(f"   BM25 Weight (α): {best['params']['alpha']:.3f}")
        print(f"   Embedding Weight (β): {best['params']['beta']:.3f}")
        
    except Exception as e:
        print(f"Could not load best results: {e}")

if __name__ == "__main__":
    main()
