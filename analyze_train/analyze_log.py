import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
from scipy.signal import savgol_filter

def parse_training_log(log_file):
    # Initialize dictionaries to store different loss values
    losses = {
        'step_loss': [],
        'step_loss_mel': [],
        'step_loss_kl': [],
        'step_loss_duration': [],
        'step_loss_gen': [],
        'step_loss_disc': [],
        'step_loss_fmaps': [],
        'step_loss_fake_disc': [],
        'step_loss_real_disc': []
    }
    steps = []
    
    loss_pattern = r'Steps:.*?(\d+)/\d+.*?step_loss=([\d.]+).*?step_loss_disc=([\d.]+).*?step_loss_duration=([\d.]+).*?step_loss_fake_disc=([\d.]+).*?step_loss_fmaps=([\d.]+).*?step_loss_gen=([\d.]+).*?step_loss_kl=([\d.]+).*?step_loss_mel=([\d.]+).*?step_loss_real_disc=([\d.]+)'
    
    with open(log_file, 'r') as f:
        content = f.read()
        matches = re.finditer(loss_pattern, content)
        
        for match in matches:
            step = int(match.group(1))
            steps.append(step)
            
            losses['step_loss'].append(float(match.group(2)))
            losses['step_loss_disc'].append(float(match.group(3)))
            losses['step_loss_duration'].append(float(match.group(4)))
            losses['step_loss_fake_disc'].append(float(match.group(5)))
            losses['step_loss_fmaps'].append(float(match.group(6)))
            losses['step_loss_gen'].append(float(match.group(7)))
            losses['step_loss_kl'].append(float(match.group(8)))
            losses['step_loss_mel'].append(float(match.group(9)))
            losses['step_loss_real_disc'].append(float(match.group(10)))
    
    return steps, losses

def plot_losses(steps, losses, output_file):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(steps, losses['step_loss'], label='Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(steps, losses['step_loss_mel'], label='Mel Loss')
    plt.plot(steps, losses['step_loss_kl'], label='KL Loss')
    plt.plot(steps, losses['step_loss_duration'], label='Duration Loss')
    plt.title('Main Component Losses')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(steps, losses['step_loss_disc'], label='Discriminator Loss')
    plt.plot(steps, losses['step_loss_fake_disc'], label='Fake Disc Loss')
    plt.plot(steps, losses['step_loss_real_disc'], label='Real Disc Loss')
    plt.title('Discriminator Losses')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(steps, losses['step_loss_gen'], label='Generator Loss')
    plt.plot(steps, losses['step_loss_fmaps'], label='Feature Maps Loss')
    plt.title('Generator and Feature Maps Losses')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def analyze_checkpoints(df, checkpoint_interval=2500):
    """Analyze and find best checkpoints based on different metrics."""
    # Add weighted sum of important metrics
    df['weighted_score'] = (
        35 * df['step_loss_mel_smooth'] +  # Higher weight for mel loss as per config
        1.5 * df['step_loss_kl_smooth'] +  # Weight from config
        1.0 * df['step_loss_duration_smooth'] +  # Standard weight
        1.0 * df['step_loss_gen_smooth'] +  # Standard weight
        3.0 * df['step_loss_disc_smooth']   # Weight from config
    )
    
    # Only consider steps that are multiples of checkpoint_interval
    checkpoints = df[df['step'] % checkpoint_interval == 0].copy()
    
    # Find best checkpoints based on different metrics
    metrics = {
        'Overall Score': 'weighted_score',
        'Mel Loss': 'step_loss_mel_smooth',
        'Total Loss': 'step_loss_smooth',
        'Generator Loss': 'step_loss_gen_smooth'
    }
    
    best_checkpoints = {}
    for name, metric in metrics.items():
        best_idx = checkpoints[metric].idxmin()
        best_step = checkpoints.loc[best_idx, 'step']
        best_value = checkpoints.loc[best_idx, metric]
        
        # Get all losses at this step
        checkpoint_losses = {
            col: checkpoints.loc[best_idx, col] 
            for col in checkpoints.columns 
            if col.endswith('_smooth') and col != metric
        }
        
        best_checkpoints[name] = {
            'step': best_step,
            'value': best_value,
            'other_losses': checkpoint_losses
        }
    
    return best_checkpoints

def find_stability_points(df, window=5, threshold=0.01):
    """
    Find points where the training loss stabilizes.
    
    Parameters:
    - df: DataFrame with step and loss columns
    - window: Number of consecutive checkpoints to consider
    - threshold: Maximum allowed percentage change in loss to consider stable
    
    Returns:
    - Dictionary with stability points for key metrics
    """
    # We only analyze checkpoints
    checkpoint_interval = 2500
    checkpoints = df[df['step'] % checkpoint_interval == 0].copy().reset_index(drop=True)
    
    # Key metrics to analyze stability
    metrics = {
        'Overall Loss': 'step_loss_smooth', 
        'Mel Loss': 'step_loss_mel_smooth',
        'KL Loss': 'step_loss_kl_smooth',
        'Duration Loss': 'step_loss_duration_smooth',
        'Weighted Score': 'weighted_score'
    }
    
    stability_points = {}
    
    for name, metric in metrics.items():
        # Find first point where loss is stable for 'window' consecutive checkpoints
        for i in range(len(checkpoints) - window):
            values = checkpoints[metric].iloc[i:i+window].values
            base = values[0]
            
            # Skip if base value is zero or very close to zero
            if base < 0.001:
                continue
                
            # Check if all percentage changes are below threshold
            changes = [abs((val - base) / base) for val in values[1:]]
            
            if all(change <= threshold for change in changes):
                stability_points[name] = {
                    'step': checkpoints['step'].iloc[i],
                    'value': base,
                    'window_steps': checkpoints['step'].iloc[i:i+window].tolist()
                }
                break
    
    return stability_points

def print_best_checkpoints(best_checkpoints):
    """Print formatted information about best checkpoints."""
    print("\nBest Checkpoints Analysis:")
    print("=" * 80)
    
    for metric, data in best_checkpoints.items():
        print(f"\n{metric} - Best Checkpoint")
        print("-" * 40)
        print(f"Step: {data['step']}")
        print(f"Value: {data['value']:.4f}")
        print("\nOther losses at this checkpoint:")
        for loss_name, loss_value in data['other_losses'].items():
            # Clean up the loss name for display
            display_name = loss_name.replace('step_loss_', '').replace('_smooth', '')
            print(f"  {display_name:15s}: {loss_value:.4f}")

def print_stability_analysis(stability_points):
    """Print formatted information about stability points."""
    print("\nStability Analysis:")
    print("=" * 80)
    print("The model is considered stable when the loss values change less than 1% across 5 consecutive checkpoints.")
    
    if not stability_points:
        print("\nNo clear stability points detected with the current criteria.")
    else:
        for metric, data in stability_points.items():
            print(f"\n{metric} stabilizes at:")
            print("-" * 40)
            print(f"Step: {data['step']}")
            print(f"Value: {data['value']:.4f}")
            print(f"Stable across steps: {', '.join(map(str, data['window_steps']))}")

def save_to_csv(steps, losses, output_file):
    # Create DataFrame
    data = {'step': steps}
    for loss_name, values in losses.items():
        data[loss_name] = values
    
    df = pd.DataFrame(data)
    
    # Calculate moving averages for smoothed values
    window_size = 100
    for col in df.columns:
        if col != 'step':
            df[f'{col}_smooth'] = df[col].rolling(window=window_size, center=True).mean()
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    return df

def print_summary_recommendations(best_checkpoints, stability_points):
    """Print practical recommendations for model selection"""
    print("\nPractical Recommendations:")
    print("=" * 80)
    
    # Mel Loss is the most important for speech quality
    mel_best = best_checkpoints.get('Mel Loss', {}).get('step', "unknown")
    total_best = best_checkpoints.get('Total Loss', {}).get('step', "unknown")
    overall_best = best_checkpoints.get('Overall Score', {}).get('step', "unknown")
    
    # Get stability point for weighted score if available
    stability_step = stability_points.get('Weighted Score', {}).get('step', None)
    if not stability_step:
        stability_step = stability_points.get('Mel Loss', {}).get('step', None)
    
    print(f"1. For best speech quality: Use checkpoint-{mel_best}")
    print(f"2. For overall performance: Use checkpoint-{overall_best}")
    
    if stability_step:
        print(f"3. For stability: Consider any checkpoint after step {stability_step}")
    
    print("\nSelection approach:")
    print("  - Convert 2-3 of these checkpoints to inference models")
    print("  - Generate samples from each using a diverse test set")
    print("  - Conduct listening tests with native speakers")
    print("  - Pay attention to tone accuracy, syllable duration, and naturalness")

def main():
    if len(sys.argv) not in [2, 3]:
        print("Usage: python analyze_log.py <log_file> [training_id]")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    if len(sys.argv) == 3:
        training_id = sys.argv[2]
    else:
        match = re.search(r'run_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}', log_file)
        if match:
            training_id = match.group(0)
        else:
            training_id = 'unknown_run'
    
    plot_file = f'{training_id}_losses.png'
    csv_file = f'{training_id}_losses.csv'
    
    print(f"Analyzing training log: {log_file}")
    print(f"Training ID: {training_id}")
    
    # Process and create outputs
    steps, losses = parse_training_log(log_file)
    if not steps:
        print("Error: No loss data found in log file.")
        sys.exit(1)
        
    print(f"Found {len(steps)} data points from step {min(steps)} to {max(steps)}")
    
    plot_losses(steps, losses, plot_file)
    df = save_to_csv(steps, losses, csv_file)
    
    # Analyze checkpoints
    best_checkpoints = analyze_checkpoints(df)
    print_best_checkpoints(best_checkpoints)
    
    # Find stability points
    stability_points = find_stability_points(df)
    print_stability_analysis(stability_points)
    
    # Print recommendations
    print_summary_recommendations(best_checkpoints, stability_points)
    
    print(f"\nOutputs created:")
    print(f"Plot saved as: {plot_file}")
    print(f"CSV saved as: {csv_file}")

if __name__ == "__main__":
    main()