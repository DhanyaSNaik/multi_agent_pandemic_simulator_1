import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class InfectionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agents=5, max_steps=28):
        super(InfectionEnv, self).__init__()
        self.num_agents = num_agents  # Number of people in the simulation
        self.max_steps = max_steps  # Simulation duration (e.g., 4 weeks)

        # Define action space: 0=no action, 1=isolate agent X, etc.
        self.action_space = spaces.Discrete(num_agents + 1)  # One action per agent + no action

        # Observation space: [status per agent, economy]
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(num_agents + 1,), dtype=np.float32
        )

        self.people = []  # List of agent objects
        self.current_step = 0
        self.economy = 100.0  # Economic metric
        
        # Default belief parameters (can be varied per agent later)
        self.base_beliefs = {
            "base_risk": 0.5,
            "trust_alpha": 2,
            "trust_beta": 2,
            "skeptic_mode": 0.3
        }

    def reset(self, seed=None, options=None):
        # Generate unique names for agents
        random.seed(seed)
        name_pool = ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Hannah"]
        chosen_names = random.sample(name_pool, self.num_agents)
        
        # Initialize agents with slight belief variations
        self.people = [
            self.Person(name, {k: v + random.uniform(-0.1, 0.1) for k, v in self.base_beliefs.items()})
            for name in chosen_names
        ]
        
        # Start with one random infection
        patient_zero = random.choice(self.people)
        patient_zero.health = "infected"
        patient_zero.infected = True

        self.current_step = 0
        self.economy = 100.0
        return self._get_obs(), {}

    class Person:
        def __init__(self, name, beliefs):
            self.name = name
            self.health = "susceptible"
            self.infected = False
            self.recovered = False
            self.days_infected = 0
            self.isolated = False
            self.vaccinated = 0
            self.mask_usage = 0.0
            self.social_contacts = 5
            self.income_level = 1.0
            self.age_factor = np.random.uniform(0, 1)
            
            # Individual personality traits
            self.risk_tolerance = np.clip(np.random.normal(beliefs["base_risk"], 0.2), 0, 1)
            self.trust_in_gov = np.random.beta(beliefs["trust_alpha"], beliefs["trust_beta"])
            self.skepticism = np.random.triangular(0, beliefs["skeptic_mode"], 1)

        def status_code(self):
            return {"susceptible": 0, "exposed": 1, "infected": 2, "recovered": 3, "dead": 4}[self.health]

    def step(self, action):
        # Enforce policies based on action
        if action > 0:
            self.people[action - 1].isolated = True
        
        # Simulate interactions & update health
        self._simulate_interactions()
        self._update_health_status()
        self._calculate_economy()
        
        obs = self._get_obs()
        reward = self._calculate_reward(action)
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = all(p.health in ["recovered", "dead"] for p in self.people)
        
        return obs, reward, terminated, truncated, {}

    def _simulate_interactions(self):
        # Random pairwise interactions
        for _ in range(self.num_agents):
            a1, a2 = random.sample(self.people, 2)
            if not a1.isolated and not a2.isolated:
                self._attempt_transmission(a1, a2)
                self._attempt_transmission(a2, a1)

    def _attempt_transmission(self, source, target):
        if source.health == "infected" and target.health == "susceptible":
            base_risk = 0.3
            protection = (source.mask_usage * 0.6 + target.mask_usage * 0.4 + target.vaccinated * 0.3)
            if random.random() < base_risk * (1 - protection):
                target.health = "exposed"

    def _update_health_status(self):
        for p in self.people:
            if p.health == "exposed" and random.random() < 0.3:
                p.health = "infected"
                p.infected = True
            elif p.health == "infected":
                p.days_infected += 1
                if p.days_infected > 14:
                    survival_chance = 0.9 - (0.2 * p.age_factor)
                    p.health = "recovered" if random.random() < survival_chance else "dead"

    def _calculate_economy(self):
        total_income = sum(p.income_level * (0.2 if p.health == "dead" else 1.0) for p in self.people)
        self.economy = (total_income / self.num_agents) * 100

    def _calculate_reward(self, action):
        infected = sum(p.health == "infected" for p in self.people)
        dead = sum(p.health == "dead" for p in self.people)
        return 0.6 * self.economy - 1.5 * infected - 5.0 * dead - (0.2 if action != 0 else 0)

    def _get_obs(self):
        return np.array([p.status_code() for p in self.people] + [self.economy], dtype=np.float32)

    def render(self, mode='human'):
        print(f"\nDay {self.current_step}")
        for p in self.people:
            iso = " (Isolated)" if p.isolated else ""
            vacc = f" V{p.vaccinated}" if p.vaccinated > 0 else ""
            print(f"{p.name}: {p.health}{iso}{vacc}")
        print(f"Economy: {self.economy:.1f}%")


if __name__ == "__main__":
    env = InfectionEnv(num_agents=5)
    obs, _ = env.reset()

    for _ in range(28):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        env.render()
        print(f"Reward: {reward:.2f}")
        if done:
            break
