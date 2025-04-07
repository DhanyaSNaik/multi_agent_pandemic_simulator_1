import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

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

    def step(self, action):
        # For testing: apply the same action to all people
        # In a true multi-agent setup, action would be a list of 27-valued actions per person
        for person in self.people:
            person.step(action)  # Action is an integer 0-26, mapped internally

        self._simulate_interactions()
        self._update_health_status()

        # every 7 days, update the average
        if self.current_step % 7 == 0:
            self._update_averages()

        obs = self._get_obs()
        reward = self._calculate_reward()
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = all(p.health in ["recovered", "dead"] for p in self.people)
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        statuses = [p.status_code() for p in self.people]
        masks = [p.mask_usage for p in self.people]
        vaccines = [p.vaccinated for p in self.people]
        contacts = [p.social_contacts for p in self.people]
        return np.concatenate([statuses, masks, vaccines, contacts], dtype=np.float32)

    # ! Modify this after we make the visualizations
    # ! unrealistic to have everyone interacting
    def _simulate_interactions(self):
        pairs = [(i, j) for i in range(self.num_people) for j in range(i + 1, self.num_people)]
        random.shuffle(pairs)
        for i, j in pairs:
            a1, a2 = self.people[i], self.people[j]
            self._attempt_transmission(a1, a2)
            self._attempt_transmission(a2, a1)

    # ! use keys in a dict instead of hardcoding - RU to fix
    def _attempt_transmission(self, source, target):
        if source.health == "infected" and target.health == "susceptible":
            # ! is this R-naught?
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

    def _calculate_reward(self):
        num_infected = sum(1 for p in self.people if p.health == "infected")
        num_deaths = sum(1 for p in self.people if p.health == "dead")
        total_reward = 0
        for p in self.people:
            # ! where is social exposure
            total_reward += p.calculate_reward(num_infected, num_deaths, self.avg_mask_usage, self.avg_vaccination)
        return total_reward / self.num_people if self.num_people > 0 else 0

    def render(self, mode='human'):
        status_map = {"susceptible": "S", "exposed": "E", "infected": "I", "recovered": "R", "dead": "D"}
        print(f"\nDay {self.current_step}")
        for p in self.people:
            vacc = f" V{p.vaccinated}" if p.vaccinated > 0 else ""
            print(f"Person {p.id}: {status_map[p.health]}{vacc}")
        print(f"Economy: {self.economy:.1f}%")

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
                return random.choice(self.actions)
            return max(self.actions, key=lambda a: self.q_table.get((state, a), 0))

        def step(self, action=None):
            if action is None:
                action = self.choose_action()
            else:
                # For testing: action is an integer 0-26 from environment
                action = self.actions[action]  # Convert to tuple

            # Action is a tuple: (mask_level, contact_level, vaccine_level)
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
            max_future_q = max([self.q_table.get((next_state, a), 0) for a in self.actions])
            current_q = self.q_table.get((state, action), 0)
            self.q_table[(state, action)] = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

if __name__ == "__main__":
    decay_rate = 0.999999
    num_episodes=1000000
    gamma=0.9
    epsilon=1


    env = InfectionEnv(num_people=5)
    obs, info = env.reset()
    for _ in range(28):
        action = env.action_space.sample()  # Random policy (0-26)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"Reward: {reward:.2f}")
        if terminated or truncated:
            print("Simulation ended.")
            break