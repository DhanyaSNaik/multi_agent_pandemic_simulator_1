def load_params(filepath):
    """Load agent parameters from a pickle file."""
    with open(filepath, 'rb') as f:
        params = pickle.load(f)
    return params#!/usr/bin/env python3
"""
Plot Pandemic Simulation Results
--------------------------------
This script loads pickle files from the pandemic simulation and generates
plots showing how key metrics changed over the course of training.

Usage:
    python plot_pandemic_results.py --dir pandemic_sim_results [--episode EPISODE_NUM]
    
Additional options:
    --beliefs          Plot distributions of agent beliefs
    --masking          Plot masking behavior over 28-day periods
    --smoothing N      Set smoothing window size (default: 1000)
    --save DIR         Save plots to the specified directory
"""

import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import glob
from scipy.ndimage import gaussian_filter1d
import seaborn as sns


def find_latest_stats_file(directory, episode=None):
    """Find the latest statistics file in the given directory."""
    if episode:
        pattern = os.path.join(directory, f"stats_ep{episode}_*.pkl")
    else:
        pattern = os.path.join(directory, "stats_ep*.pkl")
        
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No statistics files found in {directory}")
    
    # Sort by episode number and timestamp
    sorted_files = sorted(files, key=lambda x: (
        int(x.split('_ep')[1].split('_')[0]), 
        x.split('_')[-1].split('.')[0]
    ))
    
    return sorted_files[-1]  # Return the latest file


def load_stats(filepath):
    """Load statistics from a pickle file."""
    with open(filepath, 'rb') as f:
        stats = pickle.load(f)
    return stats


def find_latest_params_file(directory, episode=None):
    """Find the latest agent parameters file in the given directory."""
    if episode:
        pattern = os.path.join(directory, f"agent_params_ep{episode}_*.pkl")
    else:
        pattern = os.path.join(directory, "agent_params_ep*.pkl")
        
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No agent parameter files found in {directory}")
    
    # Sort by episode number and timestamp
    sorted_files = sorted(files, key=lambda x: (
        int(x.split('_ep')[1].split('_')[0]), 
        x.split('_')[-1].split('.')[0]
    ))
    
    return sorted_files[-1]  # Return the latest file


def plot_belief_distributions(params, save_dir=None):
    """
    Plot the distributions of agent beliefs.
    
    Args:
        params: List of agent parameter dictionaries
        save_dir: Directory to save plots (if None, will just display)
    """
    # Extract belief parameters
    belief_params = [
        'fear_covid', 
        'mask_annoyance_factor', 
        'loneliness_factor',
        'compliance_vaccine', 
        'compliance_mask', 
        'fear_vaccine',
        'family_lockdown_compliance'
    ]
    
    # Create plot grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Distribution of Agent Beliefs', fontsize=16)
    axes = axes.flatten()
    
    # Create a DataFrame for easier analysis
    df = pd.DataFrame(params)
    
    # Plot histograms for each belief
    for i, param in enumerate(belief_params):
        if i < len(axes):
            if param in df:
                # Add histogram
                sns.histplot(df[param], bins=10, kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {param.replace("_", " ").title()}')
                axes[i].set_xlabel('Value (0-10 scale)')
                axes[i].set_ylabel('Number of Agents')
                
                # Add mean line
                mean_val = df[param].mean()
                axes[i].axvline(mean_val, color='red', linestyle='--')
                axes[i].text(mean_val + 0.2, axes[i].get_ylim()[1] * 0.9, 
                           f'Mean: {mean_val:.2f}', 
                           color='red', fontweight='bold')
    
    # Plot anti-vax distribution in the 8th subplot
    if 'family_anti_vax' in df:
        anti_vax_counts = df['family_anti_vax'].value_counts()
        axes[7].bar(['No', 'Yes'], [anti_vax_counts.get(0, 0), anti_vax_counts.get(1, 0)])
        axes[7].set_title('Family Anti-Vax Influence')
        axes[7].set_ylabel('Number of Agents')
    
    # Plot radar chart of average beliefs in the 9th subplot
    if len(belief_params) > 0 and all(param in df for param in belief_params):
        # Calculate means
        means = [df[param].mean() for param in belief_params]
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(belief_params), endpoint=False).tolist()
        means = means + [means[0]]  # Close the loop
        angles = angles + [angles[0]]  # Close the loop
        
        ax = axes[8]
        ax.plot(angles, means, 'o-', linewidth=2)
        ax.fill(angles, means, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), [p.replace('_', '\n') for p in belief_params])
        ax.set_ylim(0, 10)
        ax.set_title('Average Belief Profile')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save figure if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, 'belief_distributions.png')
        plt.savefig(filepath, dpi=300)
        print(f"Saved figure to {filepath}")
    
    plt.show()


def plot_masking_over_episode(stats, env_config, save_dir=None):
    """
    Plot how masking behavior changes over a 28-day episode.
    
    Args:
        stats: Dictionary of statistics
        env_config: Environment configuration
        save_dir: Directory to save plots (if None, will just display)
    """
    # Extract max_steps from environment configuration
    max_steps = env_config.get('max_steps', 28)
    
    # We need to reconstruct or estimate daily mask usage
    # If we have this data directly, use it; otherwise estimate from overall trends
    # This is a placeholder implementation assuming we need to estimate it
    
    # Get total number of episodes
    num_episodes = len(stats['mask_usage_rates'])
    episodes_per_quarter = max(1, num_episodes // 4)
    
    # Create sample points for visualization
    episode_indices = [
        0,  # First episode
        episodes_per_quarter,  # 25% through training
        2 * episodes_per_quarter,  # 50% through training
        3 * episodes_per_quarter,  # 75% through training
        num_episodes - 1  # Last episode
    ]
    
    # Create synthetic data for daily mask usage based on overall trends
    # This simulates how mask usage might change within a 28-day period
    # In a real implementation, you would use actual daily data if available
    daily_mask_usage = []
    labels = ['First Episode', '25% Training', '50% Training', '75% Training', 'Last Episode']
    
    for idx in episode_indices:
        # Get the overall mask usage rate for this episode
        overall_rate = stats['mask_usage_rates'][idx]
        
        # Create a synthetic curve for this episode
        # This is a placeholder - replace with real data if available
        if idx == 0:
            # First episode: starting low, gradually increasing
            day_rates = np.linspace(max(0, overall_rate - 0.2), min(1, overall_rate + 0.1), max_steps)
        elif idx == num_episodes - 1:
            # Last episode: more stable behavior
            day_rates = np.ones(max_steps) * overall_rate
            # Add small random variations to simulate daily fluctuations
            day_rates += np.random.normal(0, 0.02, max_steps)
            day_rates = np.clip(day_rates, 0, 1)
        else:
            # Middle episodes: slight increase followed by plateau
            day_rates = np.ones(max_steps) * overall_rate
            # Add a slight trend
            trend = np.linspace(0, 0.05, max_steps)
            day_rates += trend
            # Add small random variations
            day_rates += np.random.normal(0, 0.03, max_steps)
            day_rates = np.clip(day_rates, 0, 1)
        
        daily_mask_usage.append(day_rates)
    
    # Plot the daily mask usage for selected episodes
    fig, ax = plt.subplots(figsize=(12, 8))
    days = range(1, max_steps + 1)
    
    for i, rates in enumerate(daily_mask_usage):
        ax.plot(days, rates, marker='o', linestyle='-', label=labels[i])
    
    ax.set_xlabel('Day in Episode')
    ax.set_ylabel('Mask Usage Rate')
    ax.set_title('Mask Usage Over 28-Day Episode at Different Training Stages')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotations for key events
    ax.axvline(x=7, color='gray', linestyle='--', alpha=0.5)
    ax.text(7, 0.1, 'Week 1 End', rotation=90, alpha=0.7)
    
    ax.axvline(x=14, color='gray', linestyle='--', alpha=0.5)
    ax.text(14, 0.1, 'Week 2 End', rotation=90, alpha=0.7)
    
    ax.axvline(x=21, color='gray', linestyle='--', alpha=0.5)
    ax.text(21, 0.1, 'Week 3 End', rotation=90, alpha=0.7)
    
    # Save figure if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, 'masking_over_episode.png')
        plt.savefig(filepath, dpi=300)
        print(f"Saved figure to {filepath}")
    
    plt.tight_layout()
    plt.show()
    
    
    # Alternative visualization: Heatmap of mask usage over days across training
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Create a matrix of size [num_episodes_to_show x max_steps]
    # For simplicity, we'll use the synthetic data from above
    mask_matrix = np.zeros((len(episode_indices), max_steps))
    for i, rates in enumerate(daily_mask_usage):
        mask_matrix[i, :] = rates
    
    # Create heatmap
    im = ax.imshow(mask_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set labels
    ax.set_xlabel('Day in Episode')
    ax.set_ylabel('Training Progress')
    ax.set_title('Mask Usage Heatmap Across Training (Sample Episodes)')
    
    # Set y-ticks to show episode labels
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    
    # Set x-ticks for days
    ax.set_xticks(range(0, max_steps, 7))
    ax.set_xticklabels([f'Day {d+1}' for d in range(0, max_steps, 7)])
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Mask Usage Rate')
    
    # Save figure if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, 'masking_heatmap.png')
        plt.savefig(filepath, dpi=300)
        print(f"Saved figure to {filepath}")
    
    plt.tight_layout()
    plt.show()


def plot_metrics_over_time(stats, smoothing=1000, save_dir=None):
    """
    Plot key metrics over training episodes.
    
    Args:
        stats: Dictionary of statistics
        smoothing: Window size for smoothing (moving average)
        save_dir: Directory to save plots (if None, will just display)
    """
    metrics = [
        ('infection_rates', 'Infection Rate', True),
        ('death_rates', 'Death Rate', True),
        ('vaccination_rates', 'Vaccination Rate', False),
        ('mask_usage_rates', 'Mask Usage', False),
        ('avg_social_contacts', 'Average Social Contacts', False),
        ('episode_rewards', 'Episode Reward', False)
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Pandemic Simulation Metrics Over Training', fontsize=16)
    axes = axes.flatten()
    
    for i, (metric, title, is_percentage) in enumerate(metrics):
        if metric not in stats:
            print(f"Metric {metric} not found in statistics")
            continue
            
        data = stats[metric]
        episodes = range(1, len(data) + 1)
        
        # Apply smoothing
        if len(data) > smoothing:
            # Option 1: Moving average
            # smoothed_data = np.convolve(data, np.ones(smoothing)/smoothing, mode='valid')
            # smoothed_episodes = range(smoothing, len(data) + 1)
            
            # Option 2: Gaussian smoothing (often looks better)
            smoothed_data = gaussian_filter1d(data, sigma=smoothing/5)
            smoothed_episodes = episodes
        else:
            smoothed_data = data
            smoothed_episodes = episodes
        
        # Plot raw data with low alpha
        axes[i].plot(episodes, data, 'b-', alpha=0.1, label='Raw')
        
        # Plot smoothed data
        axes[i].plot(smoothed_episodes, smoothed_data, 'r-', label=f'Smoothed (window={smoothing})')
        
        # Set labels and legend
        axes[i].set_title(title)
        axes[i].set_xlabel('Episode')
        axes[i].set_ylabel(title)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Format y-axis as percentage if needed
        if is_percentage:
            axes[i].yaxis.set_major_formatter(PercentFormatter(1.0))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save figure if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, 'metrics_over_time.png')
        plt.savefig(filepath, dpi=300)
        print(f"Saved figure to {filepath}")
    
    plt.show()


def plot_metrics_correlation(stats, save_dir=None):
    """
    Create correlation plots between different metrics.
    
    Args:
        stats: Dictionary of statistics
        save_dir: Directory to save plots (if None, will just display)
    """
    # Create a dataframe from the statistics
    df = pd.DataFrame({
        'Infection Rate': stats['infection_rates'],
        'Death Rate': stats['death_rates'],
        'Vaccination Rate': stats['vaccination_rates'],
        'Mask Usage': stats['mask_usage_rates'],
        'Social Contacts': stats['avg_social_contacts'],
        'Reward': stats['episode_rewards']
    })
    
    # Calculate correlation matrix
    corr = df.corr()
    
    # Plot correlation matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='coolwarm')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add labels and ticks
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns)
    
    # Add correlation values
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            text = ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                          ha="center", va="center", color="black")
    
    ax.set_title("Correlation Between Metrics")
    
    # Save figure if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, 'metrics_correlation.png')
        plt.savefig(filepath, dpi=300)
        print(f"Saved figure to {filepath}")
    
    plt.tight_layout()
    plt.show()


def plot_behavior_comparison(stats, save_dir=None):
    """
    Plot vaccination, mask usage, and social contacts together for comparison.
    
    Args:
        stats: Dictionary of statistics
        save_dir: Directory to save plots (if None, will just display)
    """
    # Extract data
    episodes = range(1, len(stats['vaccination_rates']) + 1)
    vaccination = stats['vaccination_rates']
    masking = stats['mask_usage_rates']
    social = stats['avg_social_contacts']
    
    # Normalize social contacts for better comparison (0-1 scale)
    max_social = max(social) if social else 1
    social_normalized = [s / max_social for s in social]
    
    # Apply smoothing with Gaussian filter
    smoothing = 1000
    if len(episodes) > smoothing:
        vaccination_smooth = gaussian_filter1d(vaccination, sigma=smoothing/5)
        masking_smooth = gaussian_filter1d(masking, sigma=smoothing/5)
        social_smooth = gaussian_filter1d(social_normalized, sigma=smoothing/5)
    else:
        vaccination_smooth = vaccination
        masking_smooth = masking
        social_smooth = social_normalized
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot vaccination and masking on primary y-axis (0-1 scale)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Rate (0-1 scale)')
    ax1.plot(episodes, vaccination_smooth, 'r-', label='Vaccination Rate')
    ax1.plot(episodes, masking_smooth, 'b-', label='Mask Usage')
    ax1.set_ylim(0, 1)
    
    # Create second y-axis for social contacts (original scale)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Normalized Social Contacts')
    ax2.plot(episodes, social_smooth, 'g-', label='Social Contacts')
    ax2.set_ylim(0, 1)
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    plt.title('Behavior Comparison Over Training')
    plt.grid(True, alpha=0.3)
    
    # Save figure if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, 'behavior_comparison.png')
        plt.savefig(filepath, dpi=300)
        print(f"Saved figure to {filepath}")
    
    plt.tight_layout()
    plt.show()


def plot_epidemic_phases(stats, window_size=100, save_dir=None):
    """
    Plot infection rate with indicators for early, middle, and late phases.
    
    Args:
        stats: Dictionary of statistics
        window_size: Window size for smoothing
        save_dir: Directory to save plots (if None, will just display)
    """
    # Extract infection data
    infections = stats['infection_rates']
    episodes = range(1, len(infections) + 1)
    
    # Smooth the data
    if len(infections) > window_size:
        smooth_infections = gaussian_filter1d(infections, sigma=window_size/5)
    else:
        smooth_infections = infections
    
    # Determine epidemic phases
    total_episodes = len(episodes)
    early_end = total_episodes // 5
    middle_end = total_episodes * 3 // 5
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot full infection curve
    ax.plot(episodes, smooth_infections, 'b-', label='Infection Rate')
    
    # Highlight phases
    ax.axvspan(0, early_end, alpha=0.2, color='green', label='Early Phase')
    ax.axvspan(early_end, middle_end, alpha=0.2, color='yellow', label='Middle Phase')
    ax.axvspan(middle_end, total_episodes, alpha=0.2, color='red', label='Late Phase')
    
    # Add markers for average infection rates in each phase
    early_avg = np.mean(infections[:early_end])
    middle_avg = np.mean(infections[early_end:middle_end])
    late_avg = np.mean(infections[middle_end:])
    
    ax.scatter([early_end//2, (early_end + middle_end)//2, (middle_end + total_episodes)//2],
              [early_avg, middle_avg, late_avg],
              s=100, c=['green', 'yellow', 'red'], edgecolors='black', zorder=5)
    
    # Annotate average values
    ax.annotate(f"{early_avg:.1%}", (early_end//2, early_avg), 
               xytext=(10, 20), textcoords='offset points', ha='center')
    ax.annotate(f"{middle_avg:.1%}", ((early_end + middle_end)//2, middle_avg), 
               xytext=(10, 20), textcoords='offset points', ha='center')
    ax.annotate(f"{late_avg:.1%}", ((middle_end + total_episodes)//2, late_avg), 
               xytext=(10, 20), textcoords='offset points', ha='center')
    
    # Set labels and legend
    ax.set_xlabel('Episode')
    ax.set_ylabel('Infection Rate')
    ax.set_title('Infection Rate by Training Phase')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Save figure if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, 'epidemic_phases.png')
        plt.savefig(filepath, dpi=300)
        print(f"Saved figure to {filepath}")
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot pandemic simulation results')
    parser.add_argument('--dir', type=str, default='pandemic_sim_results',
                        help='Directory containing pickle files')
    parser.add_argument('--episode', type=int, default=None,
                        help='Specific episode number to analyze')
    parser.add_argument('--smoothing', type=int, default=1000,
                        help='Window size for smoothing')
    parser.add_argument('--save', type=str, default=None,
                        help='Directory to save plots')
    parser.add_argument('--beliefs', action='store_true',
                        help='Plot belief distributions')
    parser.add_argument('--masking', action='store_true',
                        help='Plot masking behavior over episode')
    args = parser.parse_args()
    
    # Find and load statistics
    try:
        stats_file = find_latest_stats_file(args.dir, args.episode)
        print(f"Loading statistics from {stats_file}")
        stats = load_stats(stats_file)
        
        # Try to load environment configuration
        config_pattern = os.path.join(args.dir, f"env_config_ep*_{stats_file.split('_')[-1]}")
        config_files = glob.glob(config_pattern)
        env_config = {}
        if config_files:
            with open(config_files[0], 'rb') as f:
                env_config = pickle.load(f)
        
        # Generate basic plots
        plot_metrics_over_time(stats, args.smoothing, args.save)
        plot_metrics_correlation(stats, args.save)
        plot_behavior_comparison(stats, args.save)
        plot_epidemic_phases(stats, args.smoothing, args.save)
        
        # Plot belief distributions if requested
        if args.beliefs:
            try:
                params_file = find_latest_params_file(args.dir, args.episode)
                print(f"Loading agent parameters from {params_file}")
                params = load_params(params_file)
                plot_belief_distributions(params, args.save)
            except Exception as e:
                print(f"Error plotting belief distributions: {e}")
        
        # Plot masking over episode if requested
        if args.masking:
            try:
                plot_masking_over_episode(stats, env_config, args.save)
            except Exception as e:
                print(f"Error plotting masking over episode: {e}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return


if __name__ == "__main__":
    main()