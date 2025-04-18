#!/usr/bin/env python3
"""
Pandemic Belief-Based Multi-Agent Reinforcement Learning Simulation
-------------------------------------------------------------------
This script implements a pandemic simulation using multi-agent reinforcement learning 
where agent behavior is determined by individual belief systems. The simulation divides
the population into three cohorts (Science Followers, Moderates, and Freedom Prioritizers)
with distinct belief profiles affecting their decision-making during a pandemic.

NOTE: This header and parts of the code were written with the assistance of generative AI.

Usage:
    python pandemic_cohorts.py [--profile] [--profile_output FILE] [--cohort_dist X Y Z]

Results are saved to a PNG file with a timestamp in the filename.

Authors: rootma21, VivekRkay24, saswata0502, DhanyaSNaik
Last updated: April 18th, 2025
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import argparse
import random
import time
import os
import pickle
import datetime
import cProfile
import pstats
from collections import defaultdict

class InfectionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_people=100, max_steps=28, decay_rate=0.9, epsilon=1, gamma=0.9, alpha=0.1, cohort_distribution=None):
        super(InfectionEnv, self).__init__()
        self.num_people = num_people
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(27)
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
        
        self.cohort_distribution = cohort_distribution or [0.33, 0.33, 0.34]
        
        self.cohort_profiles = [
            {  # Cohort 1: "Science Followers" - High trust in medical advice, pro-mask, pro-vaccine
                'name': 'Science Followers',
                'fear_covid': (7, 10, 1),
                'mask_annoyance_factor': (1, 5, 1),
                'loneliness_factor': (3, 7, 1),
                'compliance_vaccine': (8, 10, 0.5),
                'compliance_mask': (8, 10, 0.5),
                'fear_vaccine': (1, 3, 0.5),
                'family_lockdown_compliance': (7, 10, 1),
                'family_anti_vax_prob': 0.05
            },
            {  # Cohort 2: "Moderates" - Follow the majority, moderate views
                'name': 'Moderates',
                'fear_covid': (4, 7, 1),
                'mask_annoyance_factor': (4, 7, 1),
                'loneliness_factor': (5, 8, 1),
                'compliance_vaccine': (4, 7, 1),
                'compliance_mask': (4, 7, 1),
                'fear_vaccine': (3, 6, 1),
                'family_lockdown_compliance': (4, 7, 1),
                'family_anti_vax_prob': 0.2
            },
            {  # Cohort 3: "Freedom Prioritizers" - Value individual freedom, anti-mandate
                'name': 'Freedom Prioritizers',
                'fear_covid': (1, 4, 1),
                'mask_annoyance_factor': (7, 10, 1),
                'loneliness_factor': (7, 10, 1),
                'compliance_vaccine': (1, 4, 1),
                'compliance_mask': (1, 4, 1),
                'fear_vaccine': (7, 10, 1),
                'family_lockdown_compliance': (1, 4, 1),
                'family_anti_vax_prob': 0.7
            }
        ]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        self.people = self._create_cohort_based_population()
        
        patient_zero = random.choice(self.people)
        patient_zero.health = "infected"
        patient_zero.infected = True
        self.current_step = 0
        
        self._update_averages()
        return self._get_obs(), {}

    def _create_cohort_based_population(self):
        """Create a population based on the defined cohorts and their distributions"""
        people = []
        
        cohort_counts = [int(self.num_people * dist) for dist in self.cohort_distribution]
        cohort_counts[-1] += self.num_people - sum(cohort_counts)
        
        person_id = 0
        for cohort_idx, count in enumerate(cohort_counts):
            cohort_profile = self.cohort_profiles[cohort_idx]
            
            for _ in range(count):
                person = self.Person(
                    id=person_id,
                    cohort_id=cohort_idx,
                    cohort_name=cohort_profile['name'],
                    decay_rate=self.decay_rate,
                    epsilon=self.epsilon,
                    gamma=self.gamma,
                    alpha=self.alpha
                )
                
                self._set_cohort_specific_attributes(person, cohort_profile)
                
                people.append(person)
                person_id += 1
                
        return people

    def _set_cohort_specific_attributes(self, person, cohort_profile):
        """Set a person's attributes based on their cohort profile"""
        for attr in ['fear_covid', 'mask_annoyance_factor', 'loneliness_factor', 
                    'compliance_vaccine', 'compliance_mask', 'fear_vaccine', 
                    'family_lockdown_compliance']:
            min_val, max_val, std = cohort_profile[attr]
            mean = (min_val + max_val) / 2
            val = int(np.clip(np.random.normal(mean, std), min_val, max_val))
            setattr(person, attr, val)
        
        person.family_anti_vax = np.random.choice(
            [0, 1], 
            p=[1 - cohort_profile['family_anti_vax_prob'], cohort_profile['family_anti_vax_prob']]
        )
        
        if cohort_profile['name'] == 'Science Followers':
            person.mask_usage = np.random.uniform(0.5, 0.8)
        elif cohort_profile['name'] == 'Freedom Prioritizers':
            person.mask_usage = np.random.uniform(0.0, 0.2)
        else:
            person.mask_usage = np.random.uniform(0.2, 0.5)

    def step(self, actions=None):
        if actions is None:
            actions = [person.choose_action() for person in self.people]
        
        rewards = []
        for i, person in enumerate(self.people):
            action = actions[i] if isinstance(actions, list) else actions
            
            current_state = person.get_state()
            
            person.step(action)
            
            reward = person.calculate_reward(
                sum(1 for p in self.people if p.health == "infected"),
                sum(1 for p in self.people if p.health == "dead"),
                self.avg_mask_usage, 
                self.avg_vaccination
            )
            rewards.append(reward)
            
            next_state = person.get_state()
            person.update_q_table(action, reward, next_state)

        self._simulate_interactions()
        self._update_health_status()

        if self.current_step % 7 == 0:
            self._update_averages()

        obs = self._get_obs()
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        self.current_step += 1
        
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
            
        self._simulate_cohort_specific_interactions()
    
    def _simulate_cohort_specific_interactions(self):
        """Simulates different interaction patterns based on cohort behavior"""
        for cohort_id in range(len(self.cohort_profiles)):
            cohort_members = [p for p in self.people if p.cohort_id == cohort_id and p.health != "dead"]
            
            if len(cohort_members) >= 2:
                num_interactions = min(len(cohort_members) // 2, 5)
                for _ in range(num_interactions):
                    a1, a2 = random.sample(cohort_members, 2)
                    self._attempt_transmission(a1, a2, risk_multiplier=1.2)
                    self._attempt_transmission(a2, a1, risk_multiplier=1.2)
            
            if cohort_id == 0:
                num_infected = sum(1 for p in self.people if p.health == "infected")
                if num_infected > 0.1 * self.num_people:
                    for person in cohort_members:
                        if person.vaccinated < 2 and random.random() < 0.3:
                            person.vaccinated += 1
            
            elif cohort_id == 2:
                for person in cohort_members:
                    if random.random() < 0.3 and person.social_contacts < 8:
                        person.social_contacts += 1

    def _attempt_transmission(self, source, target, risk_multiplier=1.0):
        if source.health == "infected" and target.health == "susceptible":
            base_risk = 0.5 * risk_multiplier
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
                    p.days_infected = 0
            elif p.health == "infected":
                p.days_infected += 1
                if p.days_infected > 14:
                    base_death_rate = 0.015
                    age_factor_impact = 0.03
                    
                    if p.cohort_id == 0:
                        death_rate = base_death_rate * 0.8 + (age_factor_impact * p.age_factor * 0.8)
                    elif p.cohort_id == 2:
                        death_rate = base_death_rate * 1.2 + (age_factor_impact * p.age_factor * 1.2)
                    else:
                        death_rate = base_death_rate + (age_factor_impact * p.age_factor)
                    
                    death_rate *= max(0.3, 1 - (p.vaccinated * 0.3))
                    
                    if random.random() < death_rate:
                        p.health = "dead"
                        p.infected = False
                    else:
                        p.health = "recovered"
                        p.infected = False
                        p.recovered = True
                        p.days_recovered = 0
            elif p.health == "recovered":
                p.days_recovered += 1
                
                if p.days_recovered < 7:
                    reinfection_prob = 0.001
                else:
                    max_prob = 0.02
                    days_with_waning = p.days_recovered - 7
                    max_waning_days = 21
                    reinfection_prob = min(max_prob, max_prob * days_with_waning / max_waning_days)
                
                if p.cohort_id == 0:
                    reinfection_prob *= 0.8
                elif p.cohort_id == 2:
                    reinfection_prob *= 1.2
                
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

    def render(self, mode='human'):
        status_map = {"susceptible": "S", "exposed": "E", "infected": "I", "recovered": "R", "dead": "D"}
        print(f"\nDay {self.current_step}")
        
        cohort_stats = {}
        for i in range(len(self.cohort_profiles)):
            cohort_members = [p for p in self.people if p.cohort_id == i]
            
            susceptible = sum(1 for p in cohort_members if p.health == "susceptible")
            exposed = sum(1 for p in cohort_members if p.health == "exposed")
            infected = sum(1 for p in cohort_members if p.health == "infected")
            recovered = sum(1 for p in cohort_members if p.health == "recovered")
            dead = sum(1 for p in cohort_members if p.health == "dead")
            total = len(cohort_members)
            
            living_members = [p for p in cohort_members if p.health != "dead"]
            if living_members:
                avg_mask = np.mean([p.mask_usage for p in living_members])
                avg_vacc = np.mean([p.vaccinated for p in living_members])
                avg_contacts = np.mean([p.social_contacts for p in living_members])
            else:
                avg_mask = avg_vacc = avg_contacts = 0
                
            cohort_stats[i] = {
                'name': self.cohort_profiles[i]['name'],
                'susceptible': susceptible,
                'exposed': exposed,
                'infected': infected, 
                'recovered': recovered,
                'dead': dead,
                'total': total,
                'avg_mask': avg_mask,
                'avg_vacc': avg_vacc,
                'avg_contacts': avg_contacts
            }
            
        for i, stats in cohort_stats.items():
            print(f"Cohort {i+1}: {stats['name']}")
            print(f"  S:{stats['susceptible']} E:{stats['exposed']} I:{stats['infected']} R:{stats['recovered']} D:{stats['dead']} (Total: {stats['total']})")
            print(f"  Avg Mask: {stats['avg_mask']:.2f} Avg Vacc: {stats['avg_vacc']:.2f} Avg Contacts: {stats['avg_contacts']:.2f}")
            
        print(f"Economy: {self.economy:.1f}%")
        
    def decay_epsilon(self):
        for person in self.people:
            person.epsilon *= person.decay_rate
            
    def get_cohort_statistics(self):
        """Return detailed statistics for each cohort"""
        cohort_stats = {}
        
        for i in range(len(self.cohort_profiles)):
            cohort_members = [p for p in self.people if p.cohort_id == i]
            living_members = [p for p in cohort_members if p.health != "dead"]
            
            if not living_members:
                continue
                
            stats = {
                'name': self.cohort_profiles[i]['name'],
                'total': len(cohort_members),
                'living': len(living_members),
                'susceptible': sum(1 for p in cohort_members if p.health == "susceptible"),
                'exposed': sum(1 for p in cohort_members if p.health == "exposed"),
                'infected': sum(1 for p in cohort_members if p.health == "infected"),
                'recovered': sum(1 for p in cohort_members if p.health == "recovered"),
                'dead': sum(1 for p in cohort_members if p.health == "dead"),
                'avg_mask_usage': np.mean([p.mask_usage for p in living_members]),
                'avg_vaccination': np.mean([p.vaccinated for p in living_members]),
                'avg_social_contacts': np.mean([p.social_contacts for p in living_members]),
                'fear_covid': np.mean([p.fear_covid for p in living_members]),
                'mask_annoyance': np.mean([p.mask_annoyance_factor for p in living_members]),
                'compliance_vaccine': np.mean([p.compliance_vaccine for p in living_members]),
                'compliance_mask': np.mean([p.compliance_mask for p in living_members]),
                'fear_vaccine': np.mean([p.fear_vaccine for p in living_members]),
            }
            
            cohort_stats[i] = stats
            
        return cohort_stats

    class Person:
        def __init__(self, id, cohort_id, cohort_name, decay_rate, epsilon, gamma, alpha):
            self.id = id
            self.cohort_id = cohort_id
            self.cohort_name = cohort_name
            self.health = "susceptible"
            self.infected = False
            self.recovered = False
            self.days_infected = 0
            self.days_recovered = 0
            self.vaccinated = 0
            self.mask_usage = 0.0
            self.social_contacts = 5
            self.age_factor = np.random.uniform(0, 1)
            
            self.fear_covid = 5
            self.mask_annoyance_factor = 5
            self.loneliness_factor = 5
            self.compliance_vaccine = 5
            self.compliance_mask = 5
            self.fear_vaccine = 5
            self.family_lockdown_compliance = 5
            self.family_anti_vax = 0

            self.q_table = defaultdict(float)
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.decay_rate = decay_rate
            self.actions = [
                (m, c, v) for m in range(3) for c in range(3) for v in range(3)
            ]

        def status_code(self):
            return {"susceptible": 0, "exposed": 1, "infected": 2, "recovered": 3, "dead": 4}[self.health]

        def get_state(self):
            return (self.health, self.mask_usage, self.vaccinated, self.social_contacts)

        def choose_action(self):
            state = self.get_state()
            if random.uniform(0, 1) < self.epsilon:
                return random.choice(range(27))
            
            best_action = 0
            best_value = float('-inf')
            for i in range(27):
                q_value = self.q_table[(state, i)]
                if q_value > best_value:
                    best_value = q_value
                    best_action = i
            return best_action

        def step(self, action=None):
            if action is None:
                action = self.choose_action()
                
            if isinstance(action, int):
                mask_delta, contact_level, vaccine_level = self.actions[action]
            else:
                mask_delta, contact_level, vaccine_level = action

            if self.cohort_id == 0:
                mask_compliance_factor = self.compliance_mask / 10 * 1.2
                lockdown_compliance_factor = self.family_lockdown_compliance / 10 * 1.2
                vaccine_compliance_factor = self.compliance_vaccine / 10 * (1 - self.family_anti_vax) * 1.2
            elif self.cohort_id == 2:
                mask_compliance_factor = self.compliance_mask / 10 * 0.8
                lockdown_compliance_factor = self.family_lockdown_compliance / 10 * 0.8
                vaccine_compliance_factor = self.compliance_vaccine / 10 * (1 - self.family_anti_vax) * 0.8
            else:
                mask_compliance_factor = self.compliance_mask / 10
                lockdown_compliance_factor = self.family_lockdown_compliance / 10
                vaccine_compliance_factor = self.compliance_vaccine / 10 * (1 - self.family_anti_vax)

            if mask_delta == 0:
                self.mask_usage = max(0.0, self.mask_usage - 0.1 * (1 - mask_compliance_factor))
            elif mask_delta == 1:
                pass
            elif mask_delta == 2:
                self.mask_usage = min(1.0, self.mask_usage + 0.2 * mask_compliance_factor)

            if contact_level == 0:
                self.social_contacts += int(2 * (1 - lockdown_compliance_factor))
            elif contact_level == 1:
                pass
            elif contact_level == 2:
                self.social_contacts = max(0, self.social_contacts - int(1 * lockdown_compliance_factor))

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

            if self.cohort_id == 0:
                infection_penalty *= 1.2
                mask_annoyance_penalty *= 0.8
                social_penalty *= 1.2
                vaccination_benefit *= 1.2
                
            elif self.cohort_id == 2:
                infection_penalty *= 0.8
                mask_annoyance_penalty *= 1.2  
                loneliness_penalty *= 1.2
                vaccination_fear_penalty *= 1.2
                social_penalty *= 0.8
            
            
            return (
                infection_penalty + staying_susceptible_penalty + overall_health_penalty +
                mask_annoyance_penalty + mask_usage_reward + mask_compliance_reward +
                vaccination_benefit + vaccination_fear_penalty + family_influence_reward +
                vaccination_usage_reward + social_penalty + lockdown_compliance_reward +
                loneliness_penalty + family_social_penalty
            )

        def update_q_table(self, action, reward, next_state):
            state = self.get_state()
            
            if isinstance(action, tuple):
                action_idx = self.actions.index(action)
            else:
                action_idx = action
                
            max_future_q = max([self.q_table[(next_state, a_idx)] for a_idx in range(27)])
            
            current_q = self.q_table[(state, action_idx)]
            self.q_table[(state, action_idx)] = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

def save_cohort_statistics(env, stats_dict, save_dir="cohort_stats"):
    """
    Save cohort-specific statistics to CSV files for easier analysis.
    
    Args:
        env: The environment with cohort data
        stats_dict: Dictionary of statistics by episode
        save_dir: Directory to save statistics
    """
    import csv
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    cohort_names = [profile['name'] for profile in env.cohort_profiles]
    
    overall_file = os.path.join(save_dir, f"cohort_overall_stats_{timestamp}.csv")
    
    last_episode = max(stats_dict['cohort_stats'].keys())
    cohort_stats = stats_dict['cohort_stats'][last_episode]
    
    with open(overall_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Cohort', 'Size', 'Susceptible', 'Exposed', 'Infected', 'Recovered', 'Dead',
                         'Mask Usage', 'Vaccination', 'Social Contacts', 'Fear Covid', 
                         'Mask Annoyance', 'Vaccine Compliance', 'Mask Compliance', 'Vaccine Fear'])
        
        for cohort_id, stats in cohort_stats.items():
            writer.writerow([
                stats['name'],
                stats['total'],
                stats['susceptible'],
                stats['exposed'],
                stats['infected'],
                stats['recovered'],
                stats['dead'],
                f"{stats['avg_mask_usage']:.2f}",
                f"{stats['avg_vaccination']:.2f}",
                f"{stats['avg_social_contacts']:.2f}",
                f"{stats['fear_covid']:.2f}",
                f"{stats['mask_annoyance']:.2f}",
                f"{stats['compliance_vaccine']:.2f}",
                f"{stats['compliance_mask']:.2f}",
                f"{stats['fear_vaccine']:.2f}"
            ])
    
    for cohort_id in range(len(cohort_names)):
        timeseries_file = os.path.join(save_dir, f"cohort_{cohort_id}_{cohort_names[cohort_id]}_{timestamp}.csv")
        
        with open(timeseries_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Susceptible', 'Exposed', 'Infected', 'Recovered', 'Dead',
                            'Mask Usage', 'Vaccination', 'Social Contacts'])
            
            for episode, episode_stats in sorted(stats_dict['cohort_stats'].items()):
                if cohort_id in episode_stats:
                    stats = episode_stats[cohort_id]
                    writer.writerow([
                        episode,
                        stats['susceptible'],
                        stats['exposed'],
                        stats['infected'],
                        stats['recovered'],
                        stats['dead'],
                        f"{stats['avg_mask_usage']:.2f}",
                        f"{stats['avg_vaccination']:.2f}",
                        f"{stats['avg_social_contacts']:.2f}"
                    ])
    
    print(f"Saved cohort statistics to {save_dir}")
    return {
        'overall': overall_file,
        'timeseries': timeseries_file
    }

def save_results(env, stats, episode, save_dir="saved_models"):
    """
    Save the Q-tables, statistics, and agent parameters to pickle files.
    
    Args:
        env: The environment instance containing agents
        stats: Dictionary of statistics to save
        episode: Current episode number
        save_dir: Directory to save results in
    """
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    q_tables = {}
    if hasattr(env, 'people') and env.people:
        q_tables = {f"agent_{i}": agent.q_table for i, agent in enumerate(env.people) 
                   if hasattr(agent, 'q_table')}
    
    q_tables_file = os.path.join(save_dir, f"q_tables_ep{episode}_{timestamp}.pkl")
    with open(q_tables_file, 'wb') as f:
        pickle.dump(q_tables, f)
    
    stats_file = os.path.join(save_dir, f"stats_ep{episode}_{timestamp}.pkl")
    with open(stats_file, 'wb') as f:
        pickle.dump(stats, f)

    agent_params = []
    if hasattr(env, 'people') and env.people:
        for agent in env.people:
            if hasattr(agent, 'id'):
                params = {
                    'id': agent.id,
                    'cohort_id': agent.cohort_id,
                    'cohort_name': agent.cohort_name,
                    'fear_covid': agent.fear_covid,
                    'mask_annoyance_factor': agent.mask_annoyance_factor,
                    'loneliness_factor': agent.loneliness_factor,
                    'compliance_vaccine': agent.compliance_vaccine,
                    'compliance_mask': agent.compliance_mask,
                    'fear_vaccine': agent.fear_vaccine,
                    'family_lockdown_compliance': agent.family_lockdown_compliance,
                    'family_anti_vax': agent.family_anti_vax,
                    'epsilon': agent.epsilon,
                    'health': agent.health,
                    'mask_usage': agent.mask_usage,
                    'vaccinated': agent.vaccinated,
                    'social_contacts': agent.social_contacts
                }
                agent_params.append(params)
    
    params_file = os.path.join(save_dir, f"agent_params_ep{episode}_{timestamp}.pkl")
    with open(params_file, 'wb') as f:
        pickle.dump(agent_params, f)
    
    env_config = {
        'num_people': env.num_people,
        'max_steps': env.max_steps,
        'decay_rate': env.decay_rate,
        'gamma': env.gamma,
        'alpha': env.alpha,
        'episode': episode,
        'cohort_distribution': env.cohort_distribution,
        'cohort_profiles': env.cohort_profiles
    }
    
    config_file = os.path.join(save_dir, f"env_config_ep{episode}_{timestamp}.pkl")
    with open(config_file, 'wb') as f:
        pickle.dump(env_config, f)
    
    print(f"Saved results at episode {episode} to {save_dir}")
    
    return {
        'q_tables': q_tables_file,
        'stats': stats_file,
        'params': params_file,
        'config': config_file
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Pandemic Simulator with Q-learning and Cohorts')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    parser.add_argument('--profile_output', type=str, default='profile_results.txt', 
                       help='File to save profiling results')
    parser.add_argument('--cohort_dist', type=float, nargs=3, default=[0.33, 0.33, 0.34],
                       help='Distribution of cohorts (Science Followers, Moderates, Freedom Prioritizers)')
    args = parser.parse_args()

    cohort_dist_sum = sum(args.cohort_dist)
    cohort_distribution = [x / cohort_dist_sum for x in args.cohort_dist]

    num_episodes = 1000
    num_people = 50
    max_steps = 28
    epsilon_start = 1.0
    epsilon_end = 0.01
    decay_rate = (epsilon_end / epsilon_start) ** (1 / num_episodes)
    gamma = 0.9
    alpha = 0.1
    save_interval = 100
    save_dir = "pandemic_sim_results"
    cohort_stats_dir = "cohort_stats"

    print(f"Training {num_people} agents with cohort-based Q-learning for {num_episodes} episodes.")
    print(f"Cohort Distribution: Science Followers={cohort_distribution[0]:.2%}, "
          f"Moderates={cohort_distribution[1]:.2%}, Freedom Prioritizers={cohort_distribution[2]:.2%}")
    print(f"Each episode represents a 28-day period. Epsilon decay rate: {decay_rate}")
    print(f"Results will be saved every {save_interval} episodes to '{save_dir}'")

    all_stats = {
        'episode_rewards': [],
        'infection_rates': [],
        'death_rates': [],
        'vaccination_rates': [],
        'mask_usage_rates': [],
        'avg_social_contacts': [],
        'cohort_stats': {}
    }

    start_time = time.time()

    env = InfectionEnv(
        num_people=num_people,
        max_steps=max_steps,
        decay_rate=decay_rate,
        epsilon=epsilon_start,
        gamma=gamma,
        alpha=alpha,
        cohort_distribution=cohort_distribution
    )

    try:
        save_results(env, all_stats, 0, save_dir)
        all_stats['cohort_stats'][0] = env.get_cohort_statistics()
        save_cohort_statistics(env, all_stats, cohort_stats_dir)
    except Exception as e:
        print(f"Warning: Could not save initial state: {e}")
        print("Continuing with training...")

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0

        for day in range(max_steps):
            actions = [person.choose_action() for person in env.people]
            obs, reward, terminated, truncated, info = env.step(actions)
            episode_reward += reward

            if terminated or truncated:
                break

        all_stats['episode_rewards'].append(episode_reward)
        num_infected = sum(1 for p in env.people if p.health == "infected")
        num_dead = sum(1 for p in env.people if p.health == "dead")
        avg_vaccination = np.mean([p.vaccinated for p in env.people if p.health != "dead"]) if any(p.health != "dead" for p in env.people) else 0
        avg_mask = env.avg_mask_usage
        avg_contacts = np.mean([p.social_contacts for p in env.people if p.health != "dead"]) if any(p.health != "dead" for p in env.people) else 0

        all_stats['infection_rates'].append(num_infected / num_people)
        all_stats['death_rates'].append(num_dead / num_people)
        all_stats['vaccination_rates'].append(avg_vaccination)
        all_stats['mask_usage_rates'].append(avg_mask)
        all_stats['avg_social_contacts'].append(avg_contacts)
        all_stats['cohort_stats'][episode + 1] = env.get_cohort_statistics()

        env.decay_epsilon()

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

        if (episode + 1) % save_interval == 0 or episode == num_episodes - 1:
            save_results(env, all_stats, episode + 1, save_dir)
            save_cohort_statistics(env, all_stats, cohort_stats_dir)

    print(f"Training complete. Total time: {time.time() - start_time:.1f} seconds")
    final_files = save_results(env, all_stats, num_episodes, save_dir)
    save_cohort_statistics(env, all_stats, cohort_stats_dir)
    print(f"Final results saved to:\n{final_files}")

    print("Final statistics (last 1000 episodes):")
    print(f"Avg Reward: {sum(all_stats['episode_rewards'][-1000:]) / 1000:.2f}")
    print(f"Avg Infection Rate: {sum(all_stats['infection_rates'][-1000:]) / 1000:.2%}")
    print(f"Avg Death Rate: {sum(all_stats['death_rates'][-1000:]) / 1000:.2%}")
    print(f"Avg Vaccination Rate: {sum(all_stats['vaccination_rates'][-1000:]) / 1000:.2f}")
    print(f"Avg Mask Usage: {sum(all_stats['mask_usage_rates'][-1000:]) / 1000:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pandemic Simulator with Q-learning and Cohorts')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    parser.add_argument('--profile_output', type=str, default='profile_results.txt', 
                       help='File to save profiling results')
    parser.add_argument('--cohort_dist', type=float, nargs=3, default=[0.33, 0.33, 0.34],
                       help='Distribution of cohorts (Science Followers, Moderates, Freedom Prioritizers)')
    args = parser.parse_args()

    if args.profile:
        print(f"Running with profiling. Results will be saved to {args.profile_output}")
        profiler = cProfile.Profile()
        profiler.enable()
        main()
        profiler.disable()
        with open(args.profile_output, 'w') as f:
            stats = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
            stats.print_stats(50)
        print(f"Profiling complete. Results saved to {args.profile_output}")
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats(20)
    else:
        main()