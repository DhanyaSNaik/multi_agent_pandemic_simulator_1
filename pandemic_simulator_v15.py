import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time
import os
import pickle
import datetime
import cProfile  # Added for profiling
import pstats    # Added for profiling analysis
from collections import defaultdict  # Added for Q-table optimization

class InfectionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_people=100, max_steps=28, decay_rate=0.9, epsilon=1, gamma=0.9, alpha=0.1):
        super(InfectionEnv, self).__init__()
        self.num_people = num_people
        self.max_steps = max_steps
        # Action space: single action for testing, but each person interprets it as one of 27 individual choices
        self.action_space = spaces.Discrete(18)  # Temporary: single action for compatibility
        # Observation space: [status, mask_usage, vaccinated, social_contacts] per person + economy
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(4 * self.num_people + 1,), dtype=np.float32
        )
        self.people = None
        self.current_step = 0
        self.economy = 100.0
        self.avg_mask_usage = 0.0
        self.avg_vaccination = 0.0
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.people = [
            self.Person(i, self.decay_rate, self.epsilon, self.gamma, self.alpha)
            for i in range(self.num_people)
        ]
        patient_zero = random.choice(self.people)
        patient_zero.health = "infected"
        patient_zero.infected = True
        self.current_step = 0
        
        self._update_averages()
        return self._get_obs(), {}

    def step(self, actions=None):
        # CHANGED: Now takes a list of actions, one per agent
        # If actions is None, each agent chooses independently
        if actions is None:
            actions = [person.choose_action() for person in self.people]
        
        # Process each agent's action individually
        rewards = []
        for i, person in enumerate(self.people):
            action = actions[i] if isinstance(actions, list) else actions
            
            # Store current state for Q-learning update
            current_state = person.get_state()
            
            # Apply action
            person.step(action)
            
            # Calculate individual reward (moved from _calculate_reward)
            reward = person.calculate_reward(
                sum(1 for p in self.people if p.health == "infected"),
                sum(1 for p in self.people if p.health == "dead"),
                self.avg_mask_usage, 
                self.avg_vaccination
            )
            rewards.append(reward)
            
            # ADDED: Update Q-table for this agent
            next_state = person.get_state()
            person.update_q_table(action, reward, next_state)

        self._simulate_interactions()
        self._update_health_status()

        # every 7 days, update the average
        if self.current_step % 7 == 0:
            self._update_averages()

        obs = self._get_obs()
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        self.current_step += 1
        
        # CHANGED: Simulation only terminates when everyone is dead OR after 28 days
        terminated = self.current_step >= self.max_steps
        truncated = all(p.health == "dead" for p in self.people)
        
        return obs, avg_reward, terminated, truncated, {}

    def _get_obs(self):
        statuses = [p.status_code() for p in self.people]
        masks = [p.mask_usage for p in self.people]
        vaccines = [p.vaccinated for p in self.people]
        contacts = [p.social_contacts for p in self.people]
        return np.concatenate([statuses, masks, vaccines, contacts], dtype=np.float32)

    def _simulate_interactions(self):
        pairs = [(i, j) for i in range(self.num_people) for j in range(i + 1, self.num_people)]
        random.shuffle(pairs)
        for i, j in pairs:
            a1, a2 = self.people[i], self.people[j]
            self._attempt_transmission(a1, a2)
            self._attempt_transmission(a2, a1)

    def _attempt_transmission(self, source, target):
        if source.health == "infected" and target.health == "susceptible":
            base_risk = 0.5
            protection = min(1,(
                source.mask_usage * 0.6 +
                target.mask_usage * 0.4 +
                target.vaccinated * 0.3
            ))
            social_exposure = min(1, target.social_contacts / 10)
            final_risk = base_risk * (1 - protection) * social_exposure
            if random.random() < max(0, min(1, final_risk)):
                target.health = "exposed"

    def _update_health_status(self):
        for p in self.people:
            if p.health == "exposed":
                if random.random() < 0.3:  # 30% chance of becoming infected after exposure
                    p.health = "infected"
                    p.infected = True
                    p.days_infected = 0
            elif p.health == "infected":
                p.days_infected += 1
                if p.days_infected > 14:  # After 14 days, resolve infection
                    # CHANGED: More realistic death rate (1-2% base rate, modified by age)
                    base_death_rate = 0.015  # 1.5% base death rate
                    age_factor_impact = 0.03  # Up to 3% additional risk based on age
                    death_rate = base_death_rate + (age_factor_impact * p.age_factor)
                    
                    if random.random() < death_rate:
                        p.health = "dead"
                        p.infected = False
                    else:
                        p.health = "recovered"
                        p.infected = False
                        p.recovered = True
                        p.days_recovered = 0  # Start tracking days since recovery
            elif p.health == "recovered":
                # ADDED: Reinfection model based on time since recovery
                p.days_recovered += 1
                
                # Calculate reinfection probability (immunity wanes over time)
                # Strong immunity for first 7 days, then gradually decreases
                if p.days_recovered < 7:
                    reinfection_prob = 0.001  # Very low chance in first week
                else:
                    # Gradually increasing probability of becoming susceptible again
                    # By day 28, probability reaches about 2%
                    max_prob = 0.02
                    days_with_waning = p.days_recovered - 7
                    max_waning_days = 21  # after day 7, it takes 21 more days to reach max probability
                    reinfection_prob = min(max_prob, max_prob * days_with_waning / max_waning_days)
                
                if random.random() < reinfection_prob:
                    p.health = "susceptible"
                    p.recovered = False

    def _update_averages(self):
        living_people = [p for p in self.people if p.health != "dead"]
        if living_people:
            self.avg_mask_usage = np.mean([p.mask_usage for p in living_people])
            self.avg_vaccination = np.mean([p.vaccinated for p in living_people])
        else:
            self.avg_mask_usage = 0.0
            self.avg_vaccination = 0.0

    # REMOVED: _calculate_reward method as rewards are now calculated individually per agent

    def render(self, mode='human'):
        status_map = {"susceptible": "S", "exposed": "E", "infected": "I", "recovered": "R", "dead": "D"}
        print(f"\nDay {self.current_step}")
        for p in self.people:
            vacc = f" V{p.vaccinated}" if p.vaccinated > 0 else ""
            print(f"Person {p.id}: {status_map[p.health]}{vacc}")
        print(f"Economy: {self.economy:.1f}%")
        
    # ADDED: Method to decay epsilon for all agents
    def decay_epsilon(self):
        for person in self.people:
            person.epsilon *= person.decay_rate

    class Person:
        def __init__(self, id, decay_rate, epsilon, gamma, alpha):
            self.id = id
            self.health = "susceptible"
            self.infected = False
            self.recovered = False
            self.days_infected = 0
            self.days_recovered = 0  # Added to track time since recovery for reinfection model
            self.vaccinated = 0  # 0=none, 1=partial, 2=full
            self.mask_usage = 0.0  # 0 to 1
            self.social_contacts = 5
            self.age_factor = np.random.uniform(0, 1)
            self.fear_covid = int(np.clip(np.random.normal(7, 2), 0, 10))
            self.mask_annoyance_factor = int(np.clip(np.random.normal(6, 2), 0, 10))
            self.loneliness_factor = int(np.clip(np.random.normal(5, 2), 0, 10))
            self.compliance_vaccine = int(np.clip(np.random.normal(6, 2), 0, 10))
            # Belief in mask effectiveness and mandate importance (0 = complete distrust, 10 = strong belief)
            self.compliance_mask = int(np.clip(np.random.normal(6, 2), 0, 10))
            self.fear_vaccine = int(np.clip(np.random.normal(3, 1.5), 0, 10))
            self.family_lockdown_compliance = int(np.clip(np.random.normal(5, 2), 0, 10))
            self.family_anti_vax = np.random.choice([0, 1], p=[0.8, 0.2])

            # Q-learning setup: 18 actions for individual combinations
            # OPTIMIZED: Use defaultdict for faster lookup of missing keys
            self.q_table = defaultdict(float)  # Returns 0.0 for missing keys automatically
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.decay_rate = decay_rate
            # Define 18 actions as combinations of mask_level (0-2), contact_level (0-2), vaccine_level (0-1)
            self.actions = [
                (m, c, v) for m in range(3) for c in range(3) for v in range(2)
            ]  # List of tuples: (mask_level, contact_level, vaccine_level)

        def status_code(self):
            return {"susceptible": 0, "exposed": 1, "infected": 2, "recovered": 3, "dead": 4}[self.health]

        def get_state(self):
            return (self.health, self.mask_usage, self.vaccinated, self.social_contacts)

        def choose_action(self):
            state = self.get_state()
            # random action
            if random.uniform(0, 1) < self.epsilon:
                return random.choice(range(18))  # CHANGED: Return index (0-17) instead of tuple
            
            # OPTIMIZED: Find best action based on Q-values
            # Since we're using defaultdict, no need to check if key exists
            best_action = 0
            best_value = float('-inf')
            for i in range(18):  # Using simple range is faster than enumerating actions list
                q_value = self.q_table[(state, i)]  # defaultdict returns 0.0 if key doesn't exist
                if q_value > best_value:
                    best_value = q_value
                    best_action = i
            return best_action

        def step(self, action=None):
            if action is None:
                action = self.choose_action()
                
            # CHANGED: Handle both tuple actions and integer actions
            if isinstance(action, int):
                # Convert integer (0-17) to action tuple
                mask_delta, contact_level, vaccine_level = self.actions[action]
            else:
                # Action is already a tuple
                mask_delta, contact_level, vaccine_level = action

            # Mask delta dictates any changes to masking (0=decrease, 1=stay the same, 2=increase usage)
            mask_compliance_factor = self.compliance_mask / 10
            if mask_delta == 0:
                # decreasing mask usage
                self.mask_usage = max(0.0, self.mask_usage - 0.1 * (1 - mask_compliance_factor))
            elif mask_delta == 1:
                # stay the same
                pass
            elif mask_delta == 2:
                # increasing mask usage
                self.mask_usage = min(1.0, self.mask_usage + 0.2 * mask_compliance_factor)

            # Contact level (0=increase, 1=maintain, 2=decrease)
            lockdown_compliance_factor = self.family_lockdown_compliance / 10
            if contact_level == 0:
                # action to increase contacts, by either 1 or 2 people, based on lockdown compliance
                self.social_contacts += int(2 * (1 - lockdown_compliance_factor))
            elif contact_level == 1:
                pass  # Maintain current contacts
            elif contact_level == 2:
                self.social_contacts = max(0, self.social_contacts - int(1 * lockdown_compliance_factor))

            # Vaccine level (0=no change, 1=consider partial, 2=higher consideration)
            vaccine_compliance_factor = self.compliance_vaccine / 10 * (1 - self.family_anti_vax)
            if vaccine_level > 0 and self.vaccinated < 2:
                if random.random() < vaccine_level * 0.3 * vaccine_compliance_factor:
                    self.vaccinated += 1

        def calculate_reward(self, num_infected, num_deaths, avg_mask_usage, avg_vaccination):
            infection_penalty = -self.fear_covid * self.status_code()
            staying_susceptible_penalty = self.fear_covid * 0.1 * (1 - self.vaccinated)
            overall_health_penalty = -(num_infected + num_deaths)
            mask_annoyance_penalty = -self.mask_annoyance_factor * self.mask_usage
            mask_usage_diff = abs(self.mask_usage - avg_mask_usage)
            mask_usage_reward = max(0, 1 - mask_usage_diff)
            mask_compliance_reward = self.mask_usage * self.compliance_mask * 0.1
            vaccination_benefit = self.fear_covid * 0.1 * self.vaccinated
            vaccination_fear_penalty = -self.fear_vaccine * (1 - self.vaccinated)
            family_influence_reward = (1 - self.vaccinated) * self.family_anti_vax * self.fear_vaccine * 0.1
            vaccination_usage_diff = abs(self.vaccinated - avg_vaccination)
            vaccination_usage_reward = max(0, 1 - vaccination_usage_diff)
            social_penalty = -1 / (1 + self.social_contacts) * self.fear_covid
            lockdown_compliance_reward = 1 / (1 + self.social_contacts) * self.family_lockdown_compliance * 0.1
            loneliness_penalty = -self.loneliness_factor / (1 + self.social_contacts)
            family_social_penalty = -self.social_contacts * self.family_lockdown_compliance * 0.1

            return (
                infection_penalty + staying_susceptible_penalty + overall_health_penalty +
                mask_annoyance_penalty + mask_usage_reward + mask_compliance_reward +
                vaccination_benefit + vaccination_fear_penalty + family_influence_reward +
                vaccination_usage_reward + social_penalty + lockdown_compliance_reward +
                loneliness_penalty + family_social_penalty
            )

        def update_q_table(self, action, reward, next_state):
            state = self.get_state()
            
            # CHANGED: Use action indices for Q-table keys
            if isinstance(action, tuple):
                # Convert action tuple to index
                action_idx = self.actions.index(action)
            else:
                action_idx = action
                
            # OPTIMIZED: Find maximum future Q-value using indices
            # Using defaultdict means we don't need to check for key existence
            max_future_q = max([self.q_table[(next_state, a_idx)] for a_idx in range(27)])
            
            # OPTIMIZED: Use defaultdict for simpler lookup
            current_q = self.q_table[(state, action_idx)]
            self.q_table[(state, action_idx)] = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)


# ADDED: Function to save results to pickle files
def save_results(env, stats, episode, save_dir="saved_models"):
    """
    Save the Q-tables, statistics, and agent parameters to pickle files.
    
    Args:
        env: The environment instance containing agents
        stats: Dictionary of statistics to save
        episode: Current episode number
        save_dir: Directory to save results in
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save Q-tables (ensure env.people exists and has q_table attribute)
    q_tables = {}
    if hasattr(env, 'people') and env.people:
        q_tables = {f"agent_{i}": agent.q_table for i, agent in enumerate(env.people) 
                   if hasattr(agent, 'q_table')}
    
    q_tables_file = os.path.join(save_dir, f"q_tables_ep{episode}_{timestamp}.pkl")
    with open(q_tables_file, 'wb') as f:
        pickle.dump(q_tables, f)
    
    # Save statistics
    stats_file = os.path.join(save_dir, f"stats_ep{episode}_{timestamp}.pkl")
    with open(stats_file, 'wb') as f:
        pickle.dump(stats, f)
    
    # Save agent parameters
    agent_params = []
    if hasattr(env, 'people') and env.people:
        for agent in env.people:
            if all(hasattr(agent, attr) for attr in ['id', 'fear_covid', 'mask_annoyance_factor']):
                params = {
                    'id': agent.id,
                    'fear_covid': agent.fear_covid,
                    'mask_annoyance_factor': agent.mask_annoyance_factor,
                    'loneliness_factor': agent.loneliness_factor,
                    'compliance_vaccine': agent.compliance_vaccine,
                    'compliance_mask': agent.compliance_mask,
                    'fear_vaccine': agent.fear_vaccine,
                    'family_lockdown_compliance': agent.family_lockdown_compliance,
                    'family_anti_vax': agent.family_anti_vax,
                    'epsilon': agent.epsilon
                }
                agent_params.append(params)
    
    params_file = os.path.join(save_dir, f"agent_params_ep{episode}_{timestamp}.pkl")
    with open(params_file, 'wb') as f:
        pickle.dump(agent_params, f)
    
    # Save environment configuration
    env_config = {
        'num_people': env.num_people,
        'max_steps': env.max_steps,
        'decay_rate': env.decay_rate,
        'gamma': env.gamma,
        'alpha': env.alpha,
        'episode': episode
    }
    
    config_file = os.path.join(save_dir, f"env_config_ep{episode}_{timestamp}.pkl")
    with open(config_file, 'wb') as f:
        pickle.dump(env_config, f)
    
    print(f"Saved results at episode {episode} to {save_dir}")
    
    # Return the filenames for reference
    return {
        'q_tables': q_tables_file,
        'stats': stats_file,
        'params': params_file,
        'config': config_file
    }

# ADDED: Function to run training with profiling
def run_training_with_profiling():
    """
    Run the main training routine with profiling enabled.
    This function separates the training logic for profiling purposes.
    """
    # Parameters for Q-learning
    num_episodes = 1000  # 100,000 episodes as requested
    num_people = 50       # 100 agents as mentioned in your requirements
    max_steps = 28         # Each episode is a 28-day period
    epsilon_start = 1.0    # Start with full exploration
    epsilon_end = 0.01     # End with minimal exploration
    decay_rate = (epsilon_end / epsilon_start) ** (1 / num_episodes)  # Calculate decay rate
    gamma = 0.9            # Discount factor
    alpha = 0.1            # Learning rate
    
    # ADDED: Parameters for saving results
    save_interval = 100   # Save results every 10,000 episodes
    save_dir = "pandemic_sim_results"  # Directory to save results
    
    print(f"Training {num_people} agents with independent Q-learning for {num_episodes} episodes.")
    print(f"Each episode represents a 28-day period. Epsilon decay rate: {decay_rate}")
    print(f"Results will be saved every {save_interval} episodes to '{save_dir}'")
    
    # ADDED: Dictionary to store all statistics
    all_stats = {
        'episode_rewards': [],
        'infection_rates': [],
        'death_rates': [],
        'vaccination_rates': [],
        'mask_usage_rates': [],
        'avg_social_contacts': []
    }
    
    # Start timer
    start_time = time.time()
    
    # Create environment before saving initial state
    env = InfectionEnv(num_people=num_people, max_steps=max_steps, 
                      decay_rate=decay_rate, epsilon=epsilon_start,
                      gamma=gamma, alpha=alpha)
    
    # ADDED: Save the initial state AFTER environment is created
    try:
        save_results(env, all_stats, 0, save_dir)
    except Exception as e:
        print(f"Warning: Could not save initial state: {e}")
        print("Continuing with training...")
    
    # Training loop
    for episode in range(num_episodes):
        # Reset environment for new episode
        obs, info = env.reset()
        
        # Track statistics for this episode
        episode_reward = 0
        
        # Run one complete episode (28 days)
        for day in range(max_steps):
            # Each agent independently chooses its action based on its own Q-table
            actions = [person.choose_action() for person in env.people]
            
            # Take step in environment with these actions
            obs, reward, terminated, truncated, info = env.step(actions)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        # Collect statistics at end of episode
        all_stats['episode_rewards'].append(episode_reward)
        
        # Calculate rates at the end of the episode
        num_infected = sum(1 for p in env.people if p.health == "infected")
        num_dead = sum(1 for p in env.people if p.health == "dead")
        avg_vaccination = np.mean([p.vaccinated for p in env.people if p.health != "dead"]) if env.people else 0
        avg_mask = env.avg_mask_usage
        avg_contacts = np.mean([p.social_contacts for p in env.people if p.health != "dead"]) if env.people else 0
        
        # Update all statistics in the dictionary
        all_stats['infection_rates'].append(num_infected / num_people)
        all_stats['death_rates'].append(num_dead / num_people)
        all_stats['vaccination_rates'].append(avg_vaccination)
        all_stats['mask_usage_rates'].append(avg_mask)
        all_stats['avg_social_contacts'].append(avg_contacts)
        
        # Decay epsilon for all agents at the end of each episode
        env.decay_epsilon()
        
        # Print progress
        if (episode + 1) % 1000 == 0:
            elapsed_time = time.time() - start_time
            avg_reward = sum(all_stats['episode_rewards'][-1000:]) / 1000
            avg_infections = sum(all_stats['infection_rates'][-1000:]) / 1000
            avg_deaths = sum(all_stats['death_rates'][-1000:]) / 1000
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Infections: {avg_infections:.2%} | "
                  f"Avg Deaths: {avg_deaths:.2%} | "
                  f"Elapsed: {elapsed_time:.1f}s")
        
        # ADDED: Save results at regular intervals
        if (episode + 1) % save_interval == 0 or episode == num_episodes - 1:
            save_results(env, all_stats, episode + 1, save_dir)
    
    # Training complete
    print(f"Training complete. Total time: {time.time() - start_time:.1f} seconds")
    
    # ADDED: Save final results
    final_files = save_results(env, all_stats, num_episodes, save_dir)
    print(f"Final results saved to:\n{final_files}")
    
    # Final statistics
    print("Final statistics:")
    print(f"Average reward (last 1000 episodes): {sum(all_stats['episode_rewards'][-1000:]) / 1000:.2f}")
    print(f"Average infection rate (last 1000 episodes): {sum(all_stats['infection_rates'][-1000:]) / 1000:.2%}")
    print(f"Average death rate (last 1000 episodes): {sum(all_stats['death_rates'][-1000:]) / 1000:.2%}")
    print(f"Average vaccination rate (last 1000 episodes): {sum(all_stats['vaccination_rates'][-1000:]) / 1000:.2f}")
    print(f"Average mask usage (last 1000 episodes): {sum(all_stats['mask_usage_rates'][-1000:]) / 1000:.2f}")

if __name__ == "__main__":
    # ADDED: Command line argument parsing for profiling
    import argparse
    parser = argparse.ArgumentParser(description='Pandemic Simulator with Q-learning')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    parser.add_argument('--profile_output', type=str, default='profile_results.txt', 
                       help='File to save profiling results')
    args = parser.parse_args()
    
    # Run with or without profiling
    if args.profile:
        print(f"Running with profiling. Results will be saved to {args.profile_output}")
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Run the training
        run_training_with_profiling()
        
        profiler.disable()
        
        # Save profiling results
        with open(args.profile_output, 'w') as f:
            stats = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
            stats.print_stats(50)  # Print top 50 functions by cumulative time
            
        print(f"Profiling complete. Results saved to {args.profile_output}")
        
        # Also print to console
        print("\nTop 20 functions by cumulative time:")
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats(20)
    else:
        # Run normally without profiling
        run_training_with_profiling()
