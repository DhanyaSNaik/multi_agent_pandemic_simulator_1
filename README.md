# **Multi-Agent Reinforcement Learning Pandemic Simulator**

A sophisticated agent-based model that simulates pandemic dynamics through independent reinforcement learning agents. This project models how individuals with diverse beliefs and preferences adapt their health behaviors during a pandemic.

This simulator uses Q-learning to model how 100 independent agents learn optimal strategies for navigating a pandemic environment through:

- Mask wearing behaviors
- Vaccination decisions
- Social distancing practices

Each agent has unique personality traits and beliefs that influence their decisions and learning processes, creating a realistic simulation of diverse societal responses to pandemic policies.

## Features

- **Independent Q-learning**: Each agent learns independently based on their unique experiences and beliefs
- **Realistic disease modeling**: SEIRD (Susceptible, Exposed, Infected, Recovered, Dead) epidemic states
- **Belief-based decision making**: Agents have individual parameters affecting their behavior:
  
  - Fear of COVID
  - Mask annoyance
  - Vaccine hesitancy
  - Loneliness factors
  - Compliance tendencies
  - Family influences

- **Realistic pandemic dynamics**:

  - Waning immunity and reinfection
  - Variable protection from interventions
  - Age-stratified mortality


- **Comprehensive data collection**: Tracking of infection rates, behaviors, and learning progress
