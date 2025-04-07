import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time

class InfectionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_people=3, max_steps=28, decay_rate=0.9, epsilon=1, gamma=0.9, alpha=0.1):
        super(InfectionEnv, self).__init__()
        self.num_people = num_people
        self.max_steps = max_steps
        # Action space: single action for testing, but each person interprets it as one of 27 individual choices
        self.action_space = spaces.Discrete(27)  # Temporary: single action for compatibility
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
        terminated = self.current_step >= self.max_steps
        truncated = all(p.health in ["recovered", "dead"] for p in self.people)
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
            protection = (
                source.mask_usage * 0.6 +
                target.mask_usage * 0.4 +
                target.vaccinated * 0.3
            )
            social_exposure = min(1, target.social_contacts / 10)
            final_risk = base_risk * (1 - protection) * social_exposure
            if random.random() < max(0, min(1, final_risk)):
                target.health = "exposed"

    def _update_health_status(self):
        for p in self.people:
            if p.health == "exposed":
                if random.random() < 0.3:
                    p.health = "infected"
                    p.infected = True
            elif p.health == "infected":
                p.days_infected += 1
                if p.days_infected > 14:
                    survival_chance = 0.9 - (0.2 * p.age_factor)
                    if random.random() < survival_chance:
                        p.health = "recovered"
                        p.infected = False
                        p.recovered = True
                    else:
                        p.health = "dead"
                        p.infected = False

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

            # Q-learning setup: 27 actions for individual combinations
            self.q_table = {}
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.decay_rate = decay_rate
            # Define 27 actions as combinations of mask_level (0-2), contact_level (0-2), vaccine_level (0-2)
            self.actions = [
                (m, c, v) for m in range(3) for c in range(3) for v in range(3)
            ]  # List of tuples: (mask_level, contact_level, vaccine_level)

        def status_code(self):
            return {"susceptible": 0, "exposed": 1, "infected": 2, "recovered": 3, "dead": 4}[self.health]

        def get_state(self):
            return (self.health, self.mask_usage, self.vaccinated, self.social_contacts)

        def choose_action(self):
            state = self.get_state()
            # random action
            if random.uniform(0, 1) < self.epsilon:
                return random.choice(range(27))  # CHANGED: Return index (0-26) instead of tuple
            
            # CHANGED: Find best action based on Q-values
            best_action = 0
            best_value = float('-inf')
            for i, a in enumerate(self.actions):
                q_value = self.q_table.get((state, i), 0)  # CHANGED: Use index as key
                if q_value > best_value:
                    best_value = q_value
                    best_action = i
            return best_action

        def step(self, action=None):
            if action is None:
                action = self.choose_action()
                
            # CHANGED: Handle both tuple actions and integer actions
            if isinstance(action, int):
                # Convert integer (0-26) to action tuple
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
                
            # CHANGED: Find maximum future Q-value using indices
            max_future_q = max([self.q_table.get((next_state, a_idx), 0) for a_idx in range(27)])
            
            # CHANGED: Use action index for Q-table key
            current_q = self.q_table.get((state, action_idx), 0)
            self.q_table[(state, action_idx)] = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)


# ADDED: Main function completely rewritten for independent Q-learning over 100,000 episodes
if __name__ == "__main__":
    # Parameters for Q-learning
    num_episodes = 100000  # 100,000 episodes as requested
    num_people = 100       # 100 agents as mentioned in your requirements
    max_steps = 28         # Each episode is a 28-day period
    epsilon_start = 1.0    # Start with full exploration
    epsilon_end = 0.01     # End with minimal exploration
    decay_rate = (epsilon_end / epsilon_start) ** (1 / num_episodes)  # Calculate decay rate
    gamma = 0.9            # Discount factor
    alpha = 0.1            # Learning rate
    
    print(f"Training {num_people} agents with independent Q-learning for {num_episodes} episodes.")
    print(f"Each episode represents a 28-day period. Epsilon decay rate: {decay_rate}")
    
    # Create environment
    env = InfectionEnv(num_people=num_people, max_steps=max_steps, 
                      decay_rate=decay_rate, epsilon=epsilon_start,
                      gamma=gamma, alpha=alpha)
    
    # Statistics tracking
    episode_rewards = []
    infection_rates = []
    death_rates = []
    vaccination_rates = []
    mask_usage_rates = []
    
    # Start timer
    start_time = time.time()
    
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
        episode_rewards.append(episode_reward)
        
        # Calculate rates at the end of the episode
        num_infected = sum(1 for p in env.people if p.health == "infected")
        num_dead = sum(1 for p in env.people if p.health == "dead")
        avg_vaccination = np.mean([p.vaccinated for p in env.people if p.health != "dead"]) if env.people else 0
        avg_mask = env.avg_mask_usage
        
        infection_rates.append(num_infected / num_people)
        death_rates.append(num_dead / num_people)
        vaccination_rates.append(avg_vaccination)
        mask_usage_rates.append(avg_mask)
        
        # Decay epsilon for all agents at the end of each episode
        env.decay_epsilon()
        
        # Print progress
        if (episode + 1) % 1000 == 0:
            elapsed_time = time.time() - start_time
            avg_reward = sum(episode_rewards[-1000:]) / 1000
            avg_infections = sum(infection_rates[-1000:]) / 1000
            avg_deaths = sum(death_rates[-1000:]) / 1000
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Infections: {avg_infections:.2%} | "
                  f"Avg Deaths: {avg_deaths:.2%} | "
                  f"Elapsed: {elapsed_time:.1f}s")
    
    # Training complete
    print(f"Training complete. Total time: {time.time() - start_time:.1f} seconds")
    
    # Optional: Plot results or save Q-tables for later use
    print("Final statistics:")
    print(f"Average reward (last 1000 episodes): {sum(episode_rewards[-1000:]) / 1000:.2f}")
    print(f"Average infection rate (last 1000 episodes): {sum(infection_rates[-1000:]) / 1000:.2%}")
    print(f"Average death rate (last 1000 episodes): {sum(death_rates[-1000:]) / 1000:.2%}")
    print(f"Average vaccination rate (last 1000 episodes): {sum(vaccination_rates[-1000:]) / 1000:.2f}")
    print(f"Average mask usage (last 1000 episodes): {sum(mask_usage_rates[-1000:]) / 1000:.2f}")