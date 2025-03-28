import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class InfectionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_people=3, max_steps=28):
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
        self.max_steps = max_steps # number of "days" the simulation will run
        self.economy = 100.0
        self.beliefs = {
            "base_risk": 0.5,
            "trust_alpha": 2,
            "trust_beta": 2,
            "skeptic_mode": 0.3
        }
        self.avg_mask_usage=0.0
        self.avg_vaccination=0.0

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
        #self.economy = 100.0
        return self._get_obs(), {}

    class Person:
        def __init__(self, id, beliefs, income_level):
            """ beliefs is a dictionary """
            self.id = id
            self.health = "susceptible"  # susceptible/infected/recovered/dead
            self.infected = False
            self.recovered = False
            self.days_infected = 0 # ! when are people recovered?
            self.vaccinated = 0  # 0=none, 1=partial, 2=full
            self.mask_usage = 0.0
            self.social_contacts = 5
            #self.income_level = income_level
            #self.age_factor = np.random.uniform(0, 1)  # 0=young, 1=elderly
            self.risk_tolerance = np.clip(np.random.normal(beliefs["base_risk"], 0.2), 0, 1)
            self.trust_in_gov = np.random.beta(beliefs["trust_alpha"], beliefs["trust_beta"])
            self.skepticism = np.random.triangular(0, beliefs["skeptic_mode"], 1)
            
            # Fear of getting COVID (0 = no fear, 10 = extreme fear)
            self.fear_covid = int(np.clip(np.random.normal(7, 2), 0, 10))
            
            # Annoyance from wearing masks (0 = loves masks, 10 = hates them)
            self.mask_annoyance_factor = int(np.clip(np.random.normal(6, 2), 0, 10))
            
            # Susceptibility to loneliness due to isolation (0 = never lonely, 10 = extremely lonely)
            self.loneliness_factor = int(np.clip(np.random.normal(5, 2), 0, 10))
            
            # Belief in vaccine effectiveness and mandate importance (0 = complete distrust, 10 = strong belief)
            self.compliance_vaccine = int(np.clip(np.random.normal(6, 2), 0, 10))
            
            # Belief in mask effectiveness and mandate importance (0 = complete distrust, 10 = strong belief)
            self.compliance_mask = int(np.clip(np.random.normal(6, 2), 0, 10))
            
            # Fear of vaccines (0 = no fear, 10 = extreme fear)
            self.fear_vaccine = int(np.clip(np.random.normal(3, 1.5), 0, 10))
            
            # Family's overall compliance with lockdown measures (0 = never complies, 10 = fully complies)
            self.family_lockdown_compliance = int(np.clip(np.random.normal(5, 2), 0, 10))
            
            # Is the family anti-vax? (80% chance of 0, 20% chance of 1)
            self.family_anti_vax = np.random.choice([0, 1], p=[0.8, 0.2])

            # Q- learning
            # Q-learning parameters
            self.q_table = {}  # Q-values for (state, action) pairs
            self.alpha = 0.1  # Learning rate
            self.gamma = 0.9  # Discount factor
            self.epsilon = 0.1  # Exploration rate

            # action space
            # Define possible actions
            self.actions = [
                "increase_mask", "decrease_mask", "same_mask",
                "increase_contacts", "decrease_contacts", "same_contacts",
                "increase_vaccine", "same_vaccine"
            ]

        def status_code(self):
            return {
                "susceptible": 0,
                #"exposed": 1,
                "infected": 2,
                "recovered": 3,
                "dead": 4
            }[self.health]
        
        def get_state(self):
            """Encodes the agent's current state as a tuple."""
            return (self.mask_usage, self.social_contacts, self.vaccination_status, self.health)
        
        def choose_action(self):
            """Uses epsilon-greedy to select an action."""
            state = self.get_state()
            if random.uniform(0, 1) < self.epsilon:
                return random.choice(self.actions)  # Explore
            else:
                return max(self.actions, key=lambda a: self.q_table.get((state, a), 0))  # Exploit

        
        def _calculate_reward(self):
            state = self.get_state()  # Get current state of the agent
            # Assuming state contains (health status, mask usage, social contacts, vaccination status)

            # **Infection**
            infection_penalty = -self.fear_covid * self.status_code()  # Penalty for infection based on fear of COVID
            staying_susceptible_penalty = self.fear_covid * 0.10 * (1 - self.vaccination_status)  # Staying susceptible to infection

            # **Overall Health of the System**
            # Penalty for number of new infections and deaths in the system (system-wide effects, not individual)
            overall_health_penalty = -(self.num_infected + self.num_deaths)  # This needs to be tracked in the system

            # **Masking**
            # Penalty for how annoying masks are (based on the agent's belief)
            mask_annoyance_penalty = -self.mask_annoyance_factor * self.mask_usage
            
            # Positive reward for how close the agent's mask usage is to the system's average mask usage
            mask_usage_diff = abs(self.mask_usage - self.average_mask_usage())  # Difference from the average
            mask_usage_reward = max(0, 1 - mask_usage_diff)  # Reward based on closeness to average mask usage
            
            # Compliance with the mask mandate based on how much they believe in it
            mask_compliance_reward = self.mask_usage * self.compliance_mask * 0.1  # Reward for compliance with the mask mandate

            # **Vaccination**
            # Positive reward for the belief in how bad it is to get COVID (encourages vaccination)
            vaccination_benefit = self.fear_covid * 0.1 * self.vaccination_status  # Positive reward for vaccination
            
            # Penalty for fear of being vaccinated
            vaccination_fear_penalty = -self.fear_vaccine * (1 - self.vaccination_status)
            
            # Reward for family behavior: If the family is anti-vax, the agent might get a positive reward for not vaccinating
            family_influence_reward = (1 - self.vaccination_status) * (1 - self.family_anti_vax) * self.family_fear_vaccine * 0.1
            
            # Positive reward for how close the agent's vaccination status is to the system's average vaccination rate
            vaccination_usage_diff = abs(self.vaccination_status - self.average_vaccination_usage())  # Difference from the average vaccination
            vaccination_usage_reward = max(0, 1 - vaccination_usage_diff)  # Reward based on closeness to average vaccination

            # **Social Behavior**
            # Positive reward for how few social contacts the agent has, to reduce the risk of infection
            social_penalty = -1 / (1 + self.social_contacts) * self.fear_covid  # Reward for staying isolated

            # Reward for compliance with lockdown behavior, based on the number of social contacts
            lockdown_compliance_reward = 1 / (1 + self.social_contacts) * self.family_lockdown_compliance * self.fear_covid * 0.1  # Based on family lockdown compliance
            
            # Penalty for loneliness (if they have too few social contacts)
            loneliness_penalty = -self.loneliness_factor / (1 + self.social_contacts)

            # Family behavior: Influence of family beliefs on social contacts
            family_social_penalty = -self.social_contacts * self.family_lockdown_compliance  # Penalty if family does not comply with lockdown

            # **Final Reward Calculation**
            total_reward = (
                infection_penalty + staying_susceptible_penalty + overall_health_penalty +
                mask_annoyance_penalty + mask_usage_reward + mask_compliance_reward +
                vaccination_benefit + vaccination_fear_penalty + family_influence_reward +
                vaccination_usage_reward + social_penalty + lockdown_compliance_reward +
                loneliness_penalty + family_social_penalty
            )

            return total_reward


        def step(self):
            """Takes a step in the environment based on Q-learning."""
            state = self.get_state()
            action = self.choose_action()

            # Apply action
            if action == "increase_mask":
                self.mask_usage = min(1.0, self.mask_usage + 0.1)
            elif action == "decrease_mask":
                self.mask_usage = max(0.0, self.mask_usage - 0.1)
            elif action == "increase_contacts":
                self.social_contacts += 1
            elif action == "decrease_contacts":
                self.social_contacts = max(0, self.social_contacts - 1)
            elif action == "increase_vaccine" and self.vaccination_status < 2:
                self.vaccination_status += 1

            # Calculate reward (Insert your function here)
            reward = self._calculate_reward()  # Replace with actual reward calculation

            # Get next state
            new_state = self.get_state()

            # Q-learning update rule: Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s',a')) - Q(s,a)]
            max_future_q = max([self.q_table.get((new_state, a), 0) for a in self.actions])
            self.q_table[(state, action)] = self.q_table.get((state, action), 0) + \
                self.alpha * (reward + self.gamma * max_future_q - self.q_table.get((state, action), 0))

            return new_state, reward

    # # Simulate disease spread
    # self._simulate_interactions()

    # # Update health statuses
    # self._update_health_status()

        # Update economy
        #self._calculate_economy()

    # # Compute observations and reward
    # obs = self._get_obs()
    # reward = self._calculate_reward(action)

    # # Check termination conditions
    # self.current_step += 1
    # terminated = self.current_step >= self.max_steps
    # truncated = all(p.health in ["recovered", "dead"] for p in self.people)

    # return obs, reward, terminated, truncated, {}

    def _simulate_interactions(self):
        # Random pairwise interactions
        pairs = [(i, j) for i in range(self.num_people) for j in range(i + 1, self.num_people)]
        random.shuffle(pairs)
        for i, j in pairs:
            a1 = self.people[i]
            a2 = self.people[j]
            self._attempt_transmission(a1, a2)
            self._attempt_transmission(a2, a1)

    def calculate_infection_probability(source,target):
        if source.health == "infected" and target.health == "susceptible":
            base_risk = 0.3
            protection = (
                source.mask_usage * 0.6 +
                target.mask_usage * 0.4 +
                target.vaccinated * 0.3
            ) 
            social_exposure = min(1, target.social_contacts/10)  #normalizing to [0,1]
            final_risk = base_risk * (1-protection) * social_exposure
            return max(0, min(1, final_risk))
        return 0
    
    def _attempt_transmission(self, source, target):
        infection_probability = self.calculate_infection_probability(source, target)
        if random.random() < infection_probability:
            target.health = "exposed"


    """def _attempt_transmission(self, source, target):
        if source.health == "infected" and target.health == "susceptible":
            base_risk = 0.3
            protection = (
                source.mask_usage * 0.6 +
                target.mask_usage * 0.4 +
                target.vaccinated * 0.3
            )
            if random.random() < base_risk * (1 - protection):
                target.health = "exposed"   """

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
        total = sum(p.income_level * (0.2 + 0.8 * (p.health == "susceptible")) for p in self.people if p.health!= "dead")
        self.economy = (total / self.num_people) * 100

    # def _calculate_economy(self):
    #     total = 0.0
    #     for p in self.people:
    #         if p.health == "dead":
    #             contribution = p.income_level * 0.2
    #         else:
    #             contribution = p.income_level * (0.2 + 0.8 * (p.health == "susceptible"))
    #         total += contribution
    #     self.economy = (total / self.num_people) * 100

    def calculate_average_behaviour(self): #calculates average behaviour every 14 days
        if self.current_step%14==0:
            avg_mask_usage = np.mean([p.mask_usage for p in self.people if p.health!="dead"])
            avg_vaccination = np.mean([p.vaccinated for p in self.people if p.health!="dead"])

    def _get_obs(self):
        # statuses = [p.status_code() for p in self.people]
        # return np.array(statuses + [self.economy], dtype=np.float32)
        return np.array(
            [p.status_code() for p in self.people +
            [p.mask_usage for p in self.people] +
            [p.vaccinated for p in self.people] +
            [p.social_contacts for p in self.people] +
            [self.economy], dtype = np.float32
        )

    def render(self, mode='human'):
        status_map = {
            "susceptible": "S", "exposed": "E",
            "infected": "I", "recovered": "R", "dead": "D"
        }
        print(f"\nDay {self.current_step}")
        for p in self.people:
            vacc = f" V{p.vaccinated}" if p.vaccinated > 0 else ""
            print(f"Person {p.id}: {status_map[p.health]}{vacc}")
        #: {self.economy:.1f}%")

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