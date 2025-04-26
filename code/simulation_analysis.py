import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pandemic_simulator import InfectionEnv

""" This script was assisted with generative AI tools"""

def run_and_visualize_episode(q_tables_dir, save_dir=None):
    """
    Run a 28-day episode using trained agents and visualize their behaviors.
    
    Args:
        q_tables_dir: Directory containing Q-table pickle files
        save_dir: Directory to save plots
    """
    # Find the latest Q-table file (most trained agents)
    q_files = sorted([f for f in os.listdir(q_tables_dir) if f.startswith("q_tables_ep")])
    if not q_files:
        print("No Q-table files found.")
        return
    
    latest_q_file = q_files[-1]
    print(f"Using Q-tables from {latest_q_file}")
    
    # Load the Q-tables
    with open(os.path.join(q_tables_dir, latest_q_file), 'rb') as f:
        q_tables = pickle.load(f)
    
    # Create a new environment
    env = InfectionEnv(num_people=100, max_steps=28)
    
    # Set up the agents with the trained Q-tables
    for i, person in enumerate(env.people):
        agent_key = f"agent_{i}"
        if agent_key in q_tables:
            person.q_table = q_tables[agent_key]
            # we still use some epsilon, but mostly greedy
            person.epsilon = 0.05
    
    # Reset the environment
    obs, _ = env.reset()
    
    # Data collection for each day
    daily_data = {
        'mask_usage': [],
        'social_contacts': [],
        'vaccination': [],
        'infection_rate': [],
        'deaths': []
    }
    
    # Run the episode
    for day in range(28):
        # Choose actions based on trained Q-tables
        actions = [person.choose_action() for person in env.people]
        
        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(actions)
        
        # Collect data
        living_people = [p for p in env.people if p.health != "dead"]
        num_living = len(living_people) if living_people else 1  # Avoid division by zero
        
        daily_data['mask_usage'].append(np.mean([p.mask_usage for p in living_people]) if living_people else 0)
        daily_data['social_contacts'].append(np.mean([p.social_contacts for p in living_people]) if living_people else 0)
        daily_data['vaccination'].append(np.mean([p.vaccinated for p in living_people]) if living_people else 0)
        daily_data['infection_rate'].append(sum(1 for p in env.people if p.health == "infected") / len(env.people))
        daily_data['deaths'].append(sum(1 for p in env.people if p.health == "dead") / len(env.people))
        
        # Print progress
        print(f"Day {day+1}: Infected: {daily_data['infection_rate'][-1]:.1%}, "
              f"Deaths: {daily_data['deaths'][-1]:.1%}, "
              f"Avg Mask: {daily_data['mask_usage'][-1]:.2f}, "
              f"Avg Vax: {daily_data['vaccination'][-1]:.2f}")
        
        if terminated or truncated:
            print("Episode ended early.")
            break
    
    # Create visualizations
    days = range(1, len(daily_data['mask_usage']) + 1)
    
    # Plot behavior metrics
    plt.figure(figsize=(12, 8))
    plt.plot(days, daily_data['mask_usage'], 'b-', marker='o', label='Mask Usage')
    plt.plot(days, daily_data['vaccination'], 'g-', marker='s', label='Vaccination Level')
    
    # Create second axis for social contacts (different scale)
    ax2 = plt.gca().twinx()
    ax2.plot(days, daily_data['social_contacts'], 'r-', marker='^', label='Social Contacts')
    ax2.set_ylabel('Average Social Contacts')
    
    # Add labels and legend
    plt.xlabel('Day in Episode')
    plt.ylabel('Average Value')
    plt.title('Agent Behaviors During 28-Day Episode')
    
    # Combine legends
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='best')
    plt.grid(True, alpha=0.3)
    
    # Save figure if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, 'agent_behaviors_28day.png')
        plt.savefig(filepath, dpi=300)
        print(f"Saved figure to {filepath}")
    
    plt.tight_layout()
    plt.show()
    
    # Plot disease metrics
    plt.figure(figsize=(12, 8))
    plt.plot(days, daily_data['infection_rate'], 'r-', marker='o', label='Infection Rate')
    plt.plot(days, daily_data['deaths'], 'k-', marker='x', label='Death Rate')
    
    # Add labels and legend
    plt.xlabel('Day in Episode')
    plt.ylabel('Rate')
    plt.title('Disease Progression During 28-Day Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add week markers
    for week in [7, 14, 21]:
        plt.axvline(x=week, color='gray', linestyle='--', alpha=0.5)
        plt.text(week, plt.ylim()[1]*0.9, f'Week {week//7}', 
                 rotation=90, verticalalignment='top')
    
    # Save figure if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, 'disease_progression_28day.png')
        plt.savefig(filepath, dpi=300)
        print(f"Saved figure to {filepath}")
    
    plt.tight_layout()
    plt.show()
    
    return daily_data