CLASS CitizenAgent:
    DEFINE
    PERSONALITY_TRAITS:
    risk_tolerance: 0.3  # 0=cautious, 1=risk-seeking
    trust_in_gov: 0.7  # 0=distrustful, 1=trusting
    skepticism: 0.4  # 0=accepts info, 1=conspiracy-leaning

#Creates unique behavioral profiles for each citizen
#risk_tolerance: Affects social distancing/mask use
#trust_in_gov: Determines policy compliance
#skepticism: Influences vaccine acceptance

    DEFINE STATE:
    health: "susceptible"  # [susceptible, exposed, infected, recovered, dead]
    vaccinated: 0  # 0=unvaccinated, 1=partial, 2=fully
    mask_usage: 0.0  # 0=never, 1=always
    social_contacts: 5  # Daily interactions
    income_level: 1.0  # Economic contribution

#State Tracking

#health: Tracks pandemic status
#vaccinated: Simulates partial/full vaccination effects
#mask_usage: Dynamic protection level
#social_contacts: Changes with lockdown policies
#income_level: Contributes to economic score

    PROCEDURE __init__(beliefs):
    # Initialize unique personality profile
    self.risk_tolerance = clip(normal(μ=beliefs["base_risk"], σ=0.2), 0, 1)
    self.trust_in_gov = beta_dist(α=beliefs["trust_alpha"], β=beliefs["trust_beta"])
    self.skepticism = triangular(low=0, high=1, mode=beliefs["skeptic_mode"])

#Personality Initialization

#Creates population diversity using statistical distributions:
#Normal distribution: Most people average risk-takers
#Beta distribution: Polarized trust in government
#Triangular distribution: Skepticism clusters around mode

    METHOD decide_vaccination(policy_active):
    # Vaccine decision based on beliefs + policy pressure
    if policy_active:
        compliance_chance = self.trust_in_gov * (1 - self.skepticism)
        if random() < compliance_chance:
            self.vaccinated = min(2, self.vaccinated + 1)
#Vaccination Behavior

#Combines trust and skepticism:
#High trust + low skepticism = 85% vaccination chance
#Low trust + high skepticism = 5% vaccination chance
#Simulates 2-dose regimen with possible dropouts

    METHOD choose_mask_usage(mandate_active):
    # Mask behavior combines mandate and personal choice
    if mandate_active:
        self.mask_usage = lerp(0.7, 1.0, self.trust_in_gov)  # Mandate boosts usage
    else:
        self.mask_usage = lerp(0.0, 0.3, 1 - self.risk_tolerance)

#------------------

    METHOD update_social_behavior(lockdown_level):
    # Social distancing response
    self.social_contacts = max(0,
                               round(10 * (1 - lockdown_level) * (0.5 + 0.5 * self.risk_tolerance))
#Lockdown Response

#Base contacts = 10 people/day
#lockdown_level: 0=normal, 1=full lockdown
#Risk-takers maintain 50% more contacts than cautious agents
#Full lockdown reduces contacts by 80-90%

    METHOD transmit_infection(other_agent, env_params):
    # Viral transmission during contact
    if self.health == "infected" and other_agent.health == "susceptible":
        base_risk = env_params["base_transmission"]

    # Risk mitigation factors
    protection = (self.mask_usage * 0.6 +
                  other_agent.mask_usage * 0.4 +
                  other_agent.vaccinated * 0.3)

    if random() < base_risk * (1 - protection):
        other_agent.health = "exposed"
#Viral Spread

#Source mask = 60% protection
#Receiver mask = 40% protection
#Full vaccination = 30% protection
#Combined protection reduces transmission chance

    METHOD progress_disease():
    # Individual health trajectory
    if self.health == "exposed":
        if
    random() < 0.3:  # Chance to develop symptoms
    self.health = "infected"
    self.disease_days = 0

    elif self.health == "infected":
    self.disease_days += 1
    if self.disease_days > 14:
    # Recovery/death outcome based on traits
        survival_chance = 0.9 - (0.2 * self.age_factor)
    if random() < survival_chance:
        self.health = "recovered"
    else:
        self.health = "dead"

CLASS PandemicMAS:
    DEFINE:
    agents = []  # 1M CitizenAgent instances
    policy_actions = [...]  # Same 0-8 action space
    economy = 100.0  # Aggregate economic health

    PROCEDURE
    step(central_policy_action):
    # --- Policy Enforcement ---
    mask_mandate = (action in {2, 4, 7, 8})
    lockdown_level = (action in {3, 5, 8}) * 0.8
    vaccination_drive = (action in {6, 7, 8})

    # --- Individual Agent Decisions ---
    for agent in self.agents:
        agent.choose_mask_usage(mask_mandate)
    agent.update_social_behavior(lockdown_level)
    if vaccination_drive:
        agent.decide_vaccination(policy_active=True)

#All agents adjust masks based on mandate
#Adapts social contacts to lockdown level
#Vaccination drives activate individual decisions


    # --- Disease Spread Phase ---
    for _ in range(avg_daily_contacts):
    # Random pairwise interactions
        a1, a2 = random.sample(self.agents, 2)
    a1.transmit_infection(a2, env_params)
    a2.transmit_infection(a1, env_params)

#Random pairwise interactions between agents
#Bidirectional infection chance (A→B and B→A)
#Scales with average daily contacts per person

    # --- Economic Calculation ---
    self.economy = sum(
        agent.income_level *
        (0.2 + 0.8 * (agent.health != "dead"))
        for agent in self.agents
    ) / len(self.agents)
# Still not sure about economic system

    # --- Central Reward Calculation ---
    infection_count = sum(a.health == "infected" for a in agents)
    death_count = sum(a.health == "dead" for a in agents)
    vaccination_rate = sum(a.vaccinated == 2 for a in agents)

    policy_cost=sum([
        self.lock_down_level*0.5,
        self.mask_mamdate*0.2
        self.vaccination_drive*0.1
    ])

    social_score=sum(
        3 if 3 <= agent.social_contacts <= 7 else
        -2 if agent.social_contacts > 7 else
        -2 for agent in self.agents
    ) / len(self.agents)

    reward = (0.6 * self.economy
              - 1.5 * infection_count
              - 5.0 * death_count
              -0.3 * policy_cost
              +2.0 * vaccination_rate
              +0.5 * social_score)

    return self.get_observations(), reward, done