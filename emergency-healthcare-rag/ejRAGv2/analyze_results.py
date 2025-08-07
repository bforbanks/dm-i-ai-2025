"""
Analysis and visualization utilities for hybrid retrieval optimization results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def load_results(results_file: str = "hybrid_bo_results.json"):
    """Load optimization results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def create_results_dataframe(results):
    """Convert results to pandas DataFrame for analysis."""
    data = []
    for i, res in enumerate(results['all_results']):
        row = res['params'].copy()
        row['target'] = res['target']
        row['iteration'] = i
        data.append(row)
    
    return pd.DataFrame(data)

def plot_optimization_progress(df, save_path: str = "optimization_progress.png"):
    """Plot the optimization progress over iterations."""
    plt.figure(figsize=(12, 8))
    
    # Plot target values
    plt.subplot(2, 2, 1)
    plt.plot(df['iteration'], df['target'], 'b-', alpha=0.7)
    plt.plot(df['iteration'], df['target'].cummax(), 'r-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Top-1 Accuracy')
    plt.title('Optimization Progress')
    plt.legend(['Current', 'Best So Far'])
    plt.grid(True, alpha=0.3)
    
    # Plot parameter evolution
    params = ['chunk_size_bm25', 'chunk_size_embed', 'alpha', 'beta']
    for i, param in enumerate(params):
        plt.subplot(2, 2, i+2)
        plt.scatter(df['iteration'], df[param], alpha=0.6, s=20)
        plt.xlabel('Iteration')
        plt.ylabel(param)
        plt.title(f'{param} Evolution')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_parameter_importance(df, save_path: str = "parameter_importance.png"):
    """Plot parameter importance based on correlation with target."""
    correlations = []
    for col in df.columns:
        if col not in ['target', 'iteration']:
            corr = df[col].corr(df['target'])
            correlations.append((col, abs(corr)))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    params, corrs = zip(*correlations)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(params)), corrs)
    plt.xlabel('Parameters')
    plt.ylabel('|Correlation with Target|')
    plt.title('Parameter Importance (Correlation with Top-1 Accuracy)')
    plt.xticks(range(len(params)), params, rotation=45, ha='right')
    
    # Color bars by correlation strength
    for bar, corr in zip(bars, corrs):
        if corr > 0.3:
            bar.set_color('red')
        elif corr > 0.1:
            bar.set_color('orange')
        else:
            bar.set_color('blue')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_parameter_distributions(df, save_path: str = "parameter_distributions.png"):
    """Plot distributions of parameters for successful vs unsuccessful trials."""
    # Define successful trials (top 25%)
    threshold = df['target'].quantile(0.75)
    successful = df[df['target'] >= threshold]
    unsuccessful = df[df['target'] < threshold]
    
    params = ['chunk_size_bm25', 'chunk_size_embed', 'alpha', 'beta', 'k1', 'b']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, param in enumerate(params):
        if i < len(axes):
            axes[i].hist(successful[param], alpha=0.7, label='Successful', bins=15)
            axes[i].hist(unsuccessful[param], alpha=0.7, label='Unsuccessful', bins=15)
            axes[i].set_xlabel(param)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'{param} Distribution')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_best_config(results):
    """Analyze the best configuration found."""
    best = results['best_config']
    
    print("="*60)
    print("BEST CONFIGURATION ANALYSIS")
    print("="*60)
    print(f"Top-1 Accuracy: {best['target']:.4f}")
    print("\nParameters:")
    for param, value in best['params'].items():
        print(f"  {param}: {value}")
    
    # Determine model type
    model_type = "MiniLM" if best['params']['model_selector'] < 0.5 else "ColBERT"
    print(f"\nSelected Model: {model_type}")
    
    # Analyze weights
    alpha = best['params']['alpha']
    beta = best['params']['beta']
    print(f"\nWeight Analysis:")
    print(f"  BM25 Weight (alpha): {alpha:.3f}")
    print(f"  Embedding Weight (beta): {beta:.3f}")
    print(f"  Weight Ratio (alpha/beta): {alpha/beta:.3f}" if beta > 0 else "  Weight Ratio: N/A (beta=0)")
    
    return best

def generate_summary_report(results, save_path: str = "optimization_summary.txt"):
    """Generate a comprehensive summary report."""
    df = create_results_dataframe(results)
    best = results['best_config']
    
    with open(save_path, 'w') as f:
        f.write("HYBRID RETRIEVAL OPTIMIZATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Total Trials: {len(df)}\n")
        f.write(f"Best Top-1 Accuracy: {best['target']:.4f}\n")
        f.write(f"Mean Accuracy: {df['target'].mean():.4f}\n")
        f.write(f"Std Accuracy: {df['target'].std():.4f}\n")
        f.write(f"Min Accuracy: {df['target'].min():.4f}\n")
        f.write(f"Max Accuracy: {df['target'].max():.4f}\n\n")
        
        f.write("BEST CONFIGURATION:\n")
        f.write("-"*20 + "\n")
        for param, value in best['params'].items():
            f.write(f"{param}: {value}\n")
        
        f.write(f"\nModel Type: {'MiniLM' if best['params']['model_selector'] < 0.5 else 'ColBERT'}\n")
        
        f.write("\nPARAMETER STATISTICS:\n")
        f.write("-"*20 + "\n")
        for col in df.columns:
            if col not in ['target', 'iteration']:
                f.write(f"{col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}\n")
        
        f.write("\nCORRELATIONS WITH TARGET:\n")
        f.write("-"*20 + "\n")
        for col in df.columns:
            if col not in ['target', 'iteration']:
                corr = df[col].corr(df['target'])
                f.write(f"{col}: {corr:.3f}\n")
    
    print(f"Summary report saved to: {save_path}")

def main():
    """Main analysis function."""
    try:
        # Load results
        results = load_results()
        df = create_results_dataframe(results)
        
        # Create output directory
        output_dir = Path("analysis_output")
        output_dir.mkdir(exist_ok=True)
        
        # Generate plots
        plot_optimization_progress(df, output_dir / "optimization_progress.png")
        plot_parameter_importance(df, output_dir / "parameter_importance.png")
        plot_parameter_distributions(df, output_dir / "parameter_distributions.png")
        
        # Analyze best configuration
        analyze_best_config(results)
        
        # Generate summary report
        generate_summary_report(results, output_dir / "optimization_summary.txt")
        
        print(f"\nAnalysis complete! Output files saved to: {output_dir}")
        
    except FileNotFoundError:
        print("Error: hybrid_bo_results.json not found. Run the optimization first.")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
