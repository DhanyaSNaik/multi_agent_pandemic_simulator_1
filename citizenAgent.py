import numpy as np
from scipy.stats import beta, triang
import random

class CitizenAgent:

    def __init__(self, beliefs):

        # Calculate risk tolerance using normal distribution and clipping between 0 and 1
        self.risk_tolerance = np.clip(np.random.normal(loc=beliefs["base_risk"], scale=0.2), 0, 1)

        # Calculate trust in government using beta distribution
        self.trust_in_gov = beta.rvs(a=beliefs["trust_alpha"], b=beliefs["trust_beta"])

        # Calculate skepticism using triangular distribution
        # The c parameter in scipy's triang is calculated as (mode - low) / (high - low)
        c = (beliefs["skeptic_mode"] - 0) / (1 - 0)  # low=0, high=1
        self.skepticism = triang.rvs(c=c, loc=0, scale=1)

        # Initialize state variables
        self.health = "susceptible"  # [susceptible, exposed, infected, recovered, dead]
        self.vaccinated = 0          # 0=unvaccinated, 1=partial, 2=fully
        self.mask_usage = 0.0        # 0=never, 1=always
        self.social_contacts = 5     # Daily interactions
        self.income_level = 1.0      # Economic contribution
        self.disease_days = 0
        self.age_factor = np.random.uniform(0, 1)  # 0=young, 1=elderly


    def decide_vaccination(self, policy_active):

        # Vaccine decision based on beliefs + policy pressure
        if policy_active:
            compliance_chance = self.trust_in_gov * (1 - self.skepticism)
            if random.random() < compliance_chance:
                self.vaccinated = min(2, self.vaccinated + 1)
    

    @staticmethod
    def lerp(start, end, t):
        """Linear interpolation between start and end using t (0 <= t <= 1)."""
        return start + t * (end - start)


    def choose_mask_usage(self, mandate_active):

        # Mask behavior combines mandate and personal choice
        if mandate_active:
            # Mandate boosts usage based on trust in government
            self.mask_usage = self.lerp(0.7, 1.0, self.trust_in_gov)
        else:
            # Without a mandate, risk tolerance influences mask usage
            self.mask_usage = self.lerp(0.0, 0.3, 1 - self.risk_tolerance)


    def update_social_behavior(self, lockdown_level):

        # Social distancing response
        self.social_contacts = max(
            0,
            round(10 * (1 - lockdown_level) * (0.5 + 0.5 * self.risk_tolerance))
        )
        
    
    def transmit_infection(self, other_agent, env_params):

        # Viral transmission during contact
        if self.health == "infected" and other_agent.health == "susceptible":
            base_risk = env_params["base_transmission"]

            # Risk mitigation factors
            protection = (
                self.mask_usage * 0.6 +             # Infected person's mask effectiveness
                other_agent.mask_usage * 0.4 +      # Susceptible person's mask effectiveness
                other_agent.vaccinated * 0.3        # Vaccination protection
            )

            # Determine if transmission occurs
            if random.random() < base_risk * (1 - protection):
                other_agent.health = "exposed"

    
    def progress_disease(self):

        # Individual health trajectory
        if self.health == "exposed":
            # Chance to develop symptoms and become infected
            if random.random() < 0.3:
                self.health = "infected"
                self.disease_days = 0

        elif self.health == "infected":
            # Progressing through infection
            self.disease_days += 1
            if self.disease_days > 14:
                # Recovery/death outcome based on age and other risk factors
                survival_chance = 0.9 - (0.2 * self.age_factor)
                if random.random() < survival_chance:
                    self.health = "recovered"
                else:
                    self.health = "dead"