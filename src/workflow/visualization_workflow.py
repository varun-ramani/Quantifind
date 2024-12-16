import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

def visualize_bit_depth_results(evaluation_results):
    """
    Create visualization plots for bit depth evaluation results.
    
    Args:
        evaluation_results (list): List of dictionaries containing evaluation metrics
    """
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(evaluation_results)
    
    # Set style for better visibility
    sns.set_style("whitegrid")
    
    # Create separate plots for each metric
    metrics_to_plot = ['quantized_crit', 'sig_to_complex', 'crit_ratio']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        
        # Plot both the metric and original criterion
        sns.lineplot(data=df, x='bit_depth', y=metric, marker='o', label=metric.replace('_', ' ').title())
        if metric == 'quantized_crit':
            sns.lineplot(data=df, x='bit_depth', y='original_crit', marker='o', label='Baseline')
        
        # Set labels and title
        plt.xlabel('Bit Depth')
        plt.ylabel('Criterion Value')
        plt.title(f'{metric.replace("_", " ").title()} vs Bit Depth')
        
        # Add legend
        plt.legend()
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
    
    plt.show()