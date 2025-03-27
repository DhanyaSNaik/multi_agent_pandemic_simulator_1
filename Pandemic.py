import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class InfectionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_people=3):
        super(InfectionEnv, self).__init__()
        self.num_people = num_people
        # Action space: 0 to 8 for policy combinations
        self.action_space = spaces.Discrete(9)
        # Observation space: [status_person0, status_person1, ..., economy]
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(self.num_people + 1,), dtype=np.float32
        )
        self.people = None
        self.current_step = 0
        self.max_steps = 28
        self.economy = 100.0
        self.beliefs = {
            "base_risk": 0.5,
            "trust_alpha": 2,
            "trust_beta": 2,
            "skeptic_mode": 0.3
        }

    def reset(self, seed=None, options=None):
        # Initialize people with unique IDs and random income levels
        self.people = [
            self.Person(i, self.beliefs, income_level=np.random.uniform(0.5, 1.5))
            for i in range(self.num_people)
        ]
        # Infect one person randomly
        patient_zero = random.choice(self.people)
        patient_zero.health = "infected"
        patient_zero.infected = True
        self.current_step = 0
        self.economy = 100.0
        return self._get_obs(), {}

    class Person:
        def __init__(self, id, beliefs, income_level):
            self.id = id
            self.health = "susceptible"  # susceptible/exposed/infected/recovered/dead
            self.infected = False
            self.recovered = False
            self.days_infected = 0
            self.vaccinated = 0  # 0=none, 1=partial, 2=full
            self.mask_usage = 0.0
            self.social_contacts = 5
            self.income_level = income_level
            self.age_factor = np.random.uniform(0, 1)  # 0=young, 1=elderly
            self.risk_tolerance = np.clip(np.random.normal(beliefs["base_risk"], 0.2), 0, 1)
            self.trust_in_gov = np.random.beta(beliefs["trust_alpha"], beliefs["trust_beta"])
            self.skepticism = np.random.triangular(0, beliefs["skeptic_mode"], 1)

        def status_code(self):
            return {
                "susceptible": 0,
                "exposed": 1,
                "infected": 2,
                "recovered": 3,
                "dead": 4
            }[self.health]

        def decide_vaccination(self, policy_active):
            if policy_active and not self.recovered:
                compliance_chance = self.trust_in_gov * (1 - self.skepticism)
                if random.random() < compliance_chance:
                    self.vaccinated = min(2, self.vaccinated + 1)

        def choose_mask_usage(self, mandate_active):
            if mandate_active:
                self.mask_usage = np.interp(self.trust_in_gov, [0, 1], [0.7, 1.0])
            else:
                self.mask_usage = np.interp(1 - self.risk_tolerance, [0, 1], [0.0, 0.3])

        def update_social_behavior(self, lockdown_level):
            base_contacts = 10 * (1 - lockdown_level)
            risk_modifier = 0.5 + 0.5 * self.risk_tolerance
            self.social_contacts = max(0, int(base_contacts * risk_modifier))

    def step(self, action):
        # Define policies based on action
        mask_mandate = action in {2, 4, 7, 8}
        lockdown_level = 0.8 if action in {3, 5, 8} else 0.0
        vaccination_drive = action in {6, 7, 8}

        # Update behavior
        for agent in self.people:
            if agent.health != "dead":
                agent.choose_mask_usage(mask_mandate)
                agent.update_social_behavior(lockdown_level)
                if vaccination_drive:
                    agent.decide_vaccination(True)

        # Simulate disease spread
        self._simulate_interactions()

        # Update health statuses
        self._update_health_status()

        # Update economy
        self._calculate_economy()

        #Calculate average behaviour
        self.calculate_average_behaviour()

        # Compute observations and reward
        obs = self._get_obs()
        reward = self._calculate_reward(action)

        # Check termination conditions
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = all(p.health in ["recovered", "dead"] for p in self.people)

        return obs, reward, terminated, truncated, {}

    def _simulate_interactions(self):
        # Random pairwise interactions
        pairs = [(i, j) for i in range(self.num_people) for j in range(i + 1, self.num_people)]
        random.shuffle(pairs)
        for i, j in pairs:
            a1 = self.people[i]
            a2 = self.people[j]
            self._attempt_transmission(a1, a2)
            self._attempt_transmission(a2, a1)

    def _attempt_transmission(self, source, target):
        if source.health == "infected" and target.health == "susceptible":
            base_risk = 0.3
            protection = (
                source.mask_usage * 0.6 +
                target.mask_usage * 0.4 +
                target.vaccinated * 0.3
            )
            if random.random() < base_risk * (1 - protection):
                target.health = "exposed"

    def _update_health_status(self):
        for p in self.people:
            if p.health == "exposed":
                if random.random() < 0.3:  # 30% chance to become infected
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

    def _calculate_economy(self):
        total = 0.0
        for p in self.people:
            if p.health == "dead":
                contribution = p.income_level * 0.2
            else:
                contribution = p.income_level * (0.2 + 0.8 * (p.health == "susceptible"))
            total += contribution
        self.economy = (total / self.num_people) * 100

    def calculate_average_behaviour(self): #calculates average behaviour every 14 days
        if self.current_step%14==0:
            avg_mask_usage = np.mean([p.mask_usage for p in self.people if p.health!="dead"])
            avg_vaccination = np.mean([p.vaccinated for p in self.people if p.health!="dead"])
            return avg_mask_usage, avg_vaccination
        return None, None

    def _calculate_reward(self, action):
        infected = sum(p.health == "infected" for p in self.people)
        dead = sum(p.health == "dead" for p in self.people)
        policy_cost = 0.2 if action != 0 else 0
        return 0.6 * self.economy - 1.5 * infected - 5.0 * dead - policy_cost

    def _get_obs(self):
        statuses = [p.status_code() for p in self.people]
        return np.array(statuses + [self.economy], dtype=np.float32)

    def render(self, mode='human'):
        status_map = {
            "susceptible": "S", "exposed": "E",
            "infected": "I", "recovered": "R", "dead": "D"
        }
        print(f"\nDay {self.current_step}")
        for p in self.people:
            vacc = f" V{p.vaccinated}" if p.vaccinated > 0 else ""
            print(f"Person {p.id}: {status_map[p.health]}{vacc}")
        print(f"Economy: {self.economy:.1f}%")

if __name__ == "__main__":
    env = InfectionEnv(num_people=5)  # Example with 5 people
    obs, _ = env.reset()
    for _ in range(28):
        action = env.action_space.sample()  # Random policy
        obs, reward, done, _, _ = env.step(action)
        env.render()
        print(f"Reward: {reward:.2f}")
        if done:
            break