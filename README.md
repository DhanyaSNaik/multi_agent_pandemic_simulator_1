# **Multi-Agent Reinforcement Learning Pandemic Simulator**

A sophisticated agent-based model that simulates pandemic dynamics through independent reinforcement learning agents. This project models how individuals with diverse beliefs and preferences adapt their health behaviors during a pandemic.

This simulator uses independent Q-learning to model how 100 independent agents learn optimal strategies for navigating a pandemic environment through:

- Mask wearing behaviors
- Vaccination decisions
- Social distancing practices

Each agent has unique personality traits and beliefs that influence their decisions and learning processes, creating a realistic simulation of diverse societal responses to pandemic policies.

## Features

- **Independent Q-learning**: Each agent learns independently based on their unique experiences and beliefs
- **Realistic disease modeling**: The simulator models pandemic dynamics using SEIRD framework(Susceptible, Exposed, Infected, Recovered, Dead) 
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


- **Comprehensive data collection**: Tracking of infection rates, behaviors, and learning progress every 7 days
# Directory Structure
```
multi_agent_pandemic_simulator
 └── code/
 │    ├── pandemic_simulation.py #main script
 │    ├── simulation_analysis.py # contains script to visualize simulation analysis 
 │    ├── demo_viz.py # contains script to run the demo
 │    ├── Beliefs_vis.py # contains script to visualize the beliefs
 │    ├── cohorts_viz.py #contains script to visualize agent health states by cohorts
 │    ├── pandemic_cohorts.py #runs the simulation with cohort-based initialization of beliefs
 │    └── pandemic_sim_results #contains output pickle files(saved q-tables, episode-level statistics, environment and agent configurations)
 │
 └── plots/ #contains demo and visualizations perrformed in the code directory
 │
 └── pseudo_code/ # contains pseudocode used pseudo.py
 │
 └── requirements.txt # contains list of required packages
 │
 └── README.md # This file
```
  
# How to run the project
- **Clone the repository**
```
git clone <repo_url>
cd repo_name
```
- **Install required packages**
```
pip install -r requirements.txt
```
- **Run the simulation**
  - Navigate to code directory and execute pandemic_simulator.py
      
  ```
  python pandemic_simulator.py
  ```
  - Visualize the demo
    ```
    python demo_viz.py
    ```
  - Visualize the belief states
    ```
    python Beliefs_vis.py
    ```
  - Visualize simulation analysis
    ```
    python simulation_analysis.py
    ```
  - Run simulation with cohort-based initialization of beliefs
    ```
    python pandemic_cohorts.py
    ```
  - Visualize agent health states by cohort
    ```
    python cohorts_viz.py
    ```
# Contributing
   - Contributions are welcome. Feel free to fork the repository and submit a pull request
    
# Acknowledgements
- Project developed as part of CS5100: Foundations of Artificial Intelligence
