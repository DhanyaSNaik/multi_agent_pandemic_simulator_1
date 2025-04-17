#!/usr/bin/env python3
"""
Run and Visualize Pandemic Simulation Episode
--------------------------------------------
This script loads trained agent Q-tables from pickle files and runs a single
28-day episode to visualize agent behaviors and health states.

NOTE: This header and parts of the code were written with the assistance of generative AI.

Usage:
    python demo_viz.py

Results are saved to a PNG file with a timestamp in the filename.

Authors: rootma21, VivekRkay24, saswata0502, DhanyaSNaik
Last updated: April 18th, 2025
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import glob

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pandemic_simulator_v15 import InfectionEnv

def find_latest_qtable_file(directory="pandemic_sim_results"):
    """Find the most recent Q-table pickle file in the specified directory."""
    q_table_files = glob.glob(os.path.join(directory, "q_tables_ep*.pkl"))
    
    if not q_table_files:
        raise FileNotFoundError(f"No Q-table files found in {directory}")
    
    q_table_files.sort(key=lambda x: (
        int(x.split('ep')[1].split('_')[0]),
        x.split('_')[-1].split('.')[0]
    ), reverse=True)
    
    return q_table_files[0]

def load_qtable(directory="pandemic_sim_results"):
    """Load the most recent Q-table."""
    q_table_file = find_latest_qtable_file(directory)
    print(f"Loading Q-table from: {q_table_file}")
    
    with open(q_table_file, 'rb') as f:
        q_tables = pickle.load(f)
    
    return q_tables

def setup_environment():
    """Set up the pandemic environment with default configuration."""
    print("Setting up environment with default configuration")
    env = InfectionEnv(num_people=50, max_steps=28, epsilon=0.01)
    return env

def load_qtables_into_agents(env, q_tables):
    """Load the Q-tables into the environment's agents."""
    if not q_tables:
        return
    
    for i, person in enumerate(env.people):
        agent_key = f"agent_{i}"
        if agent_key in q_tables:
            if not isinstance(person.q_table, defaultdict):
                person.q_table = defaultdict(float)
            
            for key, value in q_tables[agent_key].items():
                person.q_table[key] = value
        else:
            print(f"Warning: No Q-table found for {agent_key}")

def create_output_directory(output_dir="pandemic_behavior_gif_frames"):
    """Create directory for visualization outputs."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_action_components(action_index):
    """Convert action index (0-26) to mask, contact, and vaccine components."""
    mask_options = 3
    contact_options = 3
    vaccine_options = 3
    
    vaccine_level = action_index % vaccine_options
    contact_level = (action_index // vaccine_options) % contact_options
    mask_level = (action_index // (vaccine_options * contact_options)) % mask_options
    
    return mask_level, contact_level, vaccine_level
    
def run_episode_and_collect_data(env, q_tables):
    """Run a single episode and collect data for visualization with focus on behaviors."""
    obs, info = env.reset()
    
    load_qtables_into_agents(env, q_tables)
    
    all_data = []
    
    action_data = []
    
    new_infections = []
    previous_status = {i: person.health for i, person in enumerate(env.people)}
    previous_behaviors = {i: {
        'mask_usage': person.mask_usage,
        'social_contacts': person.social_contacts,
        'vaccinated': person.vaccinated
    } for i, person in enumerate(env.people)}
    
    for day in range(env.max_steps):
        day_data = []
        day_actions = []
        
        actions = [person.choose_action() for person in env.people]
        
        day_infections = []
        
        obs, reward, terminated, truncated, info = env.step(actions)
        
        for i, person in enumerate(env.people):
            if person.health in ["infected", "exposed"] and previous_status[i] not in ["infected", "exposed"]:
                day_infections.append(i)
            
            mask_change = person.mask_usage - previous_behaviors[i]['mask_usage']
            contacts_change = person.social_contacts - previous_behaviors[i]['social_contacts']
            vax_change = person.vaccinated - previous_behaviors[i]['vaccinated']
            
            mask_action, contact_action, vaccine_action = get_action_components(actions[i])

            agent_action = {
                'agent_id': i,
                'day': day,
                'health_status': person.health,
                'action': actions[i],
                'mask_action': mask_action,
                'contact_action': contact_action,
                'vaccine_action': vaccine_action,
                'mask_usage': person.mask_usage,
                'social_contacts': person.social_contacts,
                'vaccinated': person.vaccinated,
                'mask_change': mask_change,
                'contacts_change': contacts_change,
                'vax_change': vax_change,
                'fear_covid': person.fear_covid,
                'mask_annoyance': person.mask_annoyance_factor,
                'compliance_mask': person.compliance_mask,
                'compliance_vaccine': person.compliance_vaccine
            }
            day_actions.append(agent_action)
            
            agent_data = {
                'agent_id': i,
                'day': day,
                'health_status': person.health,
                'mask_usage': person.mask_usage,
                'vaccinated': person.vaccinated,
                'social_contacts': person.social_contacts,
                'fear_covid': person.fear_covid,
                'mask_annoyance': person.mask_annoyance_factor,
                'compliance_mask': person.compliance_mask,
                'compliance_vaccine': person.compliance_vaccine,
                'fear_vaccine': person.fear_vaccine,
                'family_lockdown_compliance': person.family_lockdown_compliance,
                'family_anti_vax': person.family_anti_vax,
                'age_factor': person.age_factor
            }
            day_data.append(agent_data)
            
            previous_status[i] = person.health
            previous_behaviors[i] = {
                'mask_usage': person.mask_usage,
                'social_contacts': person.social_contacts,
                'vaccinated': person.vaccinated
            }
        
        all_data.append(day_data)
        action_data.append(day_actions)
        new_infections.append(day_infections)
        
        if terminated or truncated:
            break
    
    return all_data, action_data, new_infections

def calculate_stats(day_data, action_data=None):
    """Calculate statistics for the given day's data."""
    status_counts = defaultdict(int)
    for agent in day_data:
        status_counts[agent['health_status']] += 1
    
    living_agents = [a for a in day_data if a['health_status'] != 'dead']
    avg_mask = np.mean([a['mask_usage'] for a in living_agents]) if living_agents else 0
    avg_vax = np.mean([a['vaccinated'] for a in living_agents]) if living_agents else 0
    avg_contacts = np.mean([a['social_contacts'] for a in living_agents]) if living_agents else 0
    
    action_stats = {}
    if action_data:
        mask_actions = [a['mask_action'] for a in action_data]
        contact_actions = [a['contact_action'] for a in action_data]
        vaccine_actions = [a['vaccine_action'] for a in action_data]
        
        mask_action_counts = [mask_actions.count(i) for i in range(3)]
        contact_action_counts = [contact_actions.count(i) for i in range(3)]
        vaccine_action_counts = [vaccine_actions.count(i) for i in range(3)]
        
        avg_mask_change = np.mean([a['mask_change'] for a in action_data if a['health_status'] != 'dead']) if action_data else 0
        avg_contacts_change = np.mean([a['contacts_change'] for a in action_data if a['health_status'] != 'dead']) if action_data else 0
        avg_vax_change = np.mean([a['vax_change'] for a in action_data if a['health_status'] != 'dead']) if action_data else 0
        
        action_stats = {
            'mask_action_counts': mask_action_counts,
            'contact_action_counts': contact_action_counts,
            'vaccine_action_counts': vaccine_action_counts,
            'avg_mask_change': avg_mask_change,
            'avg_contacts_change': avg_contacts_change,
            'avg_vax_change': avg_vax_change
        }
    
    return {
        'status_counts': status_counts,
        'avg_mask': avg_mask,
        'avg_vax': avg_vax,
        'avg_contacts': avg_contacts,
        'num_living': len(living_agents),
        'action_stats': action_stats
    }

def create_behavior_focused_visualization(all_data, action_data, new_infections, output_dir):
    """Create visualization focused on behavior changes for GIF."""
    num_days = len(all_data)
    
    first_day = all_data[0]
    num_agents = len(first_day)
    
    positions = {}
    
    for agent in first_day:
        agent_id = agent['agent_id']
        
        social_factor = agent['social_contacts'] / 10
        caution_factor = (agent['fear_covid'] / 10) * (1 - social_factor)
        
        radius = 10 * (1 - social_factor + caution_factor/2)
        angle = (agent_id / num_agents * 2 * np.pi) + (agent['family_anti_vax'] * 0.3)
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        positions[agent_id] = (x, y)
    
    status_colors = {
        'susceptible': '#9BC1BC',  # Light blue-green
        'exposed': '#F4AC45',      # Orange-yellow
        'infected': '#EE6352',     # Red-coral
        'recovered': '#57A773',    # Green
        'dead': '#89909F'          # Gray
    }
    
    behavior_history = {
        'day': [],
        'avg_mask': [],
        'avg_vax': [],
        'avg_contacts': [],
        'mask_increase': [],
        'mask_decrease': [],
        'contacts_increase': [],
        'contacts_decrease': [],
        'vax_increase': [],
        'susceptible': [],
        'exposed': [],
        'infected': [],
        'recovered': [],
        'dead': []
    }
    
    for day in range(num_days):
        day_data = all_data[day]
        day_action_data = action_data[day]
        day_infections = new_infections[day]
        
        stats = calculate_stats(day_data, day_action_data)
        
        behavior_history['day'].append(day)
        behavior_history['avg_mask'].append(stats['avg_mask'])
        behavior_history['avg_vax'].append(stats['avg_vax'])
        behavior_history['avg_contacts'].append(stats['avg_contacts'])
        
        if 'action_stats' in stats and stats['action_stats']:
            behavior_history['mask_increase'].append(stats['action_stats']['mask_action_counts'][2])
            behavior_history['mask_decrease'].append(stats['action_stats']['mask_action_counts'][0])
            behavior_history['contacts_increase'].append(stats['action_stats']['contact_action_counts'][0])
            behavior_history['contacts_decrease'].append(stats['action_stats']['contact_action_counts'][2])
            behavior_history['vax_increase'].append(stats['action_stats']['vaccine_action_counts'][1] + 
                                               stats['action_stats']['vaccine_action_counts'][2])
        
        for status in ['susceptible', 'exposed', 'infected', 'recovered', 'dead']:
            behavior_history[status].append(stats['status_counts'].get(status, 0))
        
        fig = plt.figure(figsize=(20, 12))
        gs = plt.GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1])
        
        ax_network = plt.subplot(gs[0, 0])
        ax_behaviors = plt.subplot(gs[0, 1])
        ax_actions = plt.subplot(gs[1, 0])
        ax_status = plt.subplot(gs[1, 1])
        
        for i, agent_i in enumerate(day_data):
            i_id = agent_i['agent_id']
            for j, agent_j in enumerate(day_data):
                if i < j:
                    j_id = agent_j['agent_id']
                    
                    mask_diff = abs(agent_i['mask_usage'] - agent_j['mask_usage'])
                    contact_diff = abs(agent_i['social_contacts'] - agent_j['social_contacts'])
                    vax_diff = abs(agent_i['vaccinated'] - agent_j['vaccinated'])
                    
                    if contact_diff <= 2 and mask_diff < 0.3 and vax_diff <= 1:
                        alpha = 0.15 if (agent_i['health_status'] == 'dead' or agent_j['health_status'] == 'dead') else 0.2
                        social_level = (agent_i['social_contacts'] + agent_j['social_contacts']) / 20
                        linewidth = max(0.5, social_level)
                        
                        ax_network.plot([positions[i_id][0], positions[j_id][0]], 
                                     [positions[i_id][1], positions[j_id][1]], 
                                     color='#CFCFC4', alpha=alpha, linewidth=linewidth)
        
        for infected_id in day_infections:
            x, y = positions[infected_id]
            circle = plt.Circle((x, y), 1.5, color='#EE6352', alpha=0.3, zorder=1)
            ax_network.add_patch(circle)
        
        for agent in day_data:
            agent_id = agent['agent_id']
            x, y = positions[agent_id]
            
            base_size = 300
            size_factor = 1 + (agent['social_contacts'] / 10)
            size = base_size * size_factor
            
            color = status_colors[agent['health_status']]
            
            if agent['health_status'] == 'dead':
                ax_network.scatter(x, y, s=size*0.7, color=color, alpha=0.5, edgecolor='black', linewidth=0.5)
            else:
                ax_network.scatter(x, y, s=size, color=color, edgecolor='black', linewidth=1)
                
                if agent['vaccinated'] > 0:
                    ring_size = size + 100
                    ring_color = '#57A773' if agent['vaccinated'] == 2 else '#A9C5A0'
                    ring_width = 3 if agent['vaccinated'] == 2 else 2
                    ax_network.scatter(x, y, s=ring_size, color='none', edgecolor=ring_color, linewidth=ring_width)
                
                if agent['mask_usage'] > 0.3:
                    mask_size = size * 0.4
                    mask_alpha = agent['mask_usage']
                    ax_network.scatter(x, y, s=mask_size, color='white', alpha=mask_alpha, edgecolor='none')
        
        margin = 2
        max_radius = 12
        ax_network.set_xlim(-max_radius-margin, max_radius+margin)
        ax_network.set_ylim(-max_radius-margin, max_radius+margin)
        ax_network.axis('off')
        
        ax_network.set_title(f"Agent Network - Day {day+1}", fontsize=14)
        
        current_days = behavior_history['day'][:day+1]
        
        ax_behaviors.plot(current_days, [v/2 for v in behavior_history['avg_vax'][:day+1]], 
                         color='#38B000', linewidth=3, label='Vaccination (scaled)')
        
        ax_contacts = ax_behaviors.twinx()
        ax_contacts.plot(current_days, behavior_history['avg_contacts'][:day+1], 
                        color='#FF006E', linewidth=3, label='Social Contacts')
        
        ax_behaviors.set_xlabel('Day')
        ax_behaviors.set_ylabel('Vaccination Level (Scaled)')
        ax_contacts.set_ylabel('Social Contacts', color='#FF006E')
        ax_contacts.tick_params(axis='y', labelcolor='#FF006E')
        ax_behaviors.set_title('Agent Behavior Trends Over Time', fontsize=14)
        
        ax_behaviors.set_xlim(0, num_days-1)
        ax_behaviors.set_ylim(0, 1.1)
        
        max_contacts = max(behavior_history['avg_contacts'][:day+1]) if day > 0 else 10
        ax_contacts.set_ylim(0, max(10, max_contacts * 1.2))  # Ensure enough space and at least up to 10
        
        ax_behaviors.grid(True, linestyle='--', alpha=0.3)
        
        ax_behaviors.axvline(x=day, color='black', linestyle='--', alpha=0.5)
        
        lines1, labels1 = ax_behaviors.get_legend_handles_labels()
        lines2, labels2 = ax_contacts.get_legend_handles_labels()
        ax_behaviors.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        if day > 0:
            display_days = min(5, day+1)
            recent_days = range(day+1-display_days, day+1)
            
            labels = list(range(day+1-display_days, day+1))
            contacts_increase = [behavior_history['contacts_increase'][d] for d in recent_days]
            contacts_decrease = [-behavior_history['contacts_decrease'][d] for d in recent_days]  # Negative for display
            vax_consider = [behavior_history['vax_increase'][d] for d in recent_days]
            
            bar_width = 0.4
            
            contacts_pos = np.arange(len(labels))
            vax_pos = contacts_pos + bar_width
            
            ax_actions.bar(contacts_pos, contacts_decrease, bar_width, color='#38B000', label='Decrease Contacts')
            ax_actions.bar(contacts_pos, contacts_increase, bar_width, color='#FF006E', label='Increase Contacts')
            
            ax_actions.bar(vax_pos, vax_consider, bar_width, color='#FFBE0B', label='Consider Vaccination')
            
            ax_actions.set_xlabel('Day')
            ax_actions.set_ylabel('Number of Agents')
            ax_actions.set_title('Agent Action Choices (Recent Days)', fontsize=14)
            
            ax_actions.set_xticks(contacts_pos + bar_width/2)
            ax_actions.set_xticklabels([f"Day {d+1}" for d in recent_days])
            
            ax_actions.grid(True, axis='y', linestyle='--', alpha=0.3)
            ax_actions.legend(loc='upper left')
            
            y_min, y_max = ax_actions.get_ylim()
            y_min = min(y_min, -num_agents/2)
            y_max = max(y_max, num_agents/2)
            ax_actions.set_ylim(y_min, y_max)
            
            ax_actions.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        else:
            ax_actions.text(0.5, 0.5, "Action data will appear after Day 1", 
                         ha='center', va='center', transform=ax_actions.transAxes, fontsize=12)
            ax_actions.set_title('Agent Action Choices (Recent Days)', fontsize=14)
            ax_actions.set_xlabel('Day')
            ax_actions.set_ylabel('Number of Agents')
            ax_actions.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        days = behavior_history['day'][:day+1]
        susceptible = behavior_history['susceptible'][:day+1]
        exposed = behavior_history['exposed'][:day+1]
        infected = behavior_history['infected'][:day+1]
        recovered = behavior_history['recovered'][:day+1]
        dead = behavior_history['dead'][:day+1]
        
        ax_status.stackplot(days, 
                          susceptible, 
                          exposed, 
                          infected, 
                          recovered, 
                          dead, 
                          labels=['Susceptible', 'Exposed', 'Infected', 'Recovered', 'Dead'],
                          colors=[status_colors['susceptible'], 
                                 status_colors['exposed'], 
                                 status_colors['infected'], 
                                 status_colors['recovered'], 
                                 status_colors['dead']])
        
        ax_status.set_xlabel('Day')
        ax_status.set_ylabel('Number of Agents')
        ax_status.set_title('Population Health Status Over Time', fontsize=14)
        
        ax_status.set_xlim(0, num_days-1)
        ax_status.set_ylim(0, num_agents)
        
        ax_status.grid(True, linestyle='--', alpha=0.3)
        ax_status.axvline(x=day, color='black', linestyle='--', alpha=0.5)
        ax_status.legend(loc='upper right')
        
        plt.suptitle(f"Pandemic Agent Behavior Evolution - Day {day+1}", fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        output_file = os.path.join(output_dir, f"pandemic_behavior_day_{day+1:02d}.png")
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved behavior visualization frame for Day {day+1} to {output_file}")

def main():
    """Main function to run the visualization process."""
    print("Starting pandemic behavior change visualization for GIF...")
    
    q_tables = load_qtable()
    env = setup_environment()
    output_dir = create_output_directory()
    
    try:
        all_data, action_data, new_infections = run_episode_and_collect_data(env, q_tables)
        
        create_behavior_focused_visualization(all_data, action_data, new_infections, output_dir)
        
        print(f"Visualization complete. All frames saved to {output_dir}")
        print("\nTo create a GIF, you can use a tool like ImageMagick with the command:")
        print(f"convert -delay 100 -loop 0 {output_dir}/pandemic_behavior_day_*.png pandemic_behavior.gif")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()