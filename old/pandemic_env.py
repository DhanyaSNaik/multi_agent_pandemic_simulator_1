import gymnasium as gym
from typing import Optional
import numpy as np
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete

"""
Notes:
- we want to use independent learning
- need to decide on communication strategies - https://www.restack.io/p/multi-agents-answer-openai-cat-ai#cm07aomsw007zvp4cchdzek74

"""

class MultiCitizenAgent(gym.Env):

    # set the metadata
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    
    def __init__(self , num_agents=3):
        # super allows us to call methods from a parent class within a child class
        super(MultiCitizenAgent, self).__init__()

        self.num_agents = num_agents
        action_low = np.array([0,1,0]) # can add more features but stuck to 3 for now as not sure if it works
        action_high = np.array([1,8,2])

        self.action_space = spaces.Box(low = action_low, high = action_high, dtype=np.float)
        # continuous action and observation spaces
        self.observation_space = spaces.Box(low = action_low, high = action_high, shape=(3*num_agents,), dtype=float)

        # Initaialize all agents as a list of dictionaries
        self.agents = []
        for _ in range(self.num_agents):
            agent = {
                'health' :'susceptible',  # [susceptible, exposed, infected, recovered, dead]
                'vaccinated' : 0 ,         # 0=unvaccinated, 1=partial, 2=fully
                'mask_usage' : 0.0,        # 0=never, 1=always
                'social_contacts' : 5,     # Daily interactions
                'income_level' : 1.0,      # Economic contribution
                'disease_days' : 0,
                'age_factor' : np.random.uniform(0, 1)  # 0=young, 1=elderly
            }
            self.agents.append(agent)

    def step(self, action_n):
        obs_n = list()
        reward_n = list()
        done_n = list()
        info_n = {'n': []}
        
        for i, actions in enumerate(action):
            agent = self.agents[i]

            agent['mask_usage'] = actions[0]
            agent['social_contacts'] = max(1,min(8, int(action[1])))
            agent['vaccinated'] = min(2, max(0, int(action[2])))
            
            obs_ = self._get_obs(agent)
            obs_n.apend(obs)

            reward = self.calc_reward(agent)
            reward_n.append(reward)

            done = self._is_done(agent)
            done_n.append(done)

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # Reset the environment
        pass

    def render(self, mode='human'):  
        # Render the environment
        pass

    def step(self, action_n):
        obs_n    = list()
        reward_n = list()
        done_n   = list()
        info_n   = {'n': []}
        # ...
        return obs_n, reward_n, done_n, info_n
    


def _get_obs(self):
    #return {"agent": self._agent_location, "health": self.health}
    pass


def _get_info(self):
    pass
    # return {
    #     "distance": np.linalg.norm(
    #         self._agent_location - self._target_location, ord=1
    #     )
    # }

def calc_reward(self):
    pass

def _is_done(self):
    pass
# EXAMPLE:

# def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
#     # We need the following line to seed self.np_random
#     super().reset(seed=seed)

#     # Choose the agent's location uniformly at random
#     self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

#     # We will sample the target's location randomly until it does not coincide with the agent's location
#     self._target_location = self._agent_location
#     while np.array_equal(self._target_location, self._agent_location):
#         self._target_location = self.np_random.integers(
#             0, self.size, size=2, dtype=int
#         )

#     observation = self._get_obs()
#     info = self._get_info()

#     return observation, info

# def step(self, action):
#     # Map the action (element of {0,1,2,3}) to the direction we walk in
#     direction = self._action_to_direction[action]
#     # We use `np.clip` to make sure we don't leave the grid bounds
#     self._agent_location = np.clip(
#         self._agent_location + direction, 0, self.size - 1
#     )

#     # An environment is completed if and only if the agent has reached the target
#     terminated = np.array_equal(self._agent_location, self._target_location)
#     truncated = False
#     reward = 1 if terminated else 0  # the agent is only reached at the end of the episode
#     observation = self._get_obs()
#     info = self._get_info()

#     return observation, reward, terminated, truncated, info

# gym.register(
#     id="gymnasium_env/GridWorld-v0",
#     entry_point=GridWorldEnv,
# )