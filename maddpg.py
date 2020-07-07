from ddpg_agent import BATCH_SIZE
from ddpg_agent import BUFFER_SIZE
from ddpg_agent import GAMMA

from ddpg_agent import Agent
from ddpg_agent import ReplayBuffer

import numpy as np

class maddpg:
    def __init__(self, state_size, action_size, random_seed, num_agents):
        self.action_size = action_size
        self.num_agents = num_agents
        self.agents = [Agent(state_size, action_size, random_seed) for i in range(num_agents)]
        self.shared_memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, states, actions, rewards, next_states, dones):
        for i in range(self.num_agents):
            self.agents[i].step(states[i,:], actions[i,:], rewards[i], next_states[i,:], dones[i])
        self.shared_memory.add(states, actions, rewards, next_states, dones)

        """
        if len(self.shared_memory) > BATCH_SIZE:
            for i in range(self.num_agents):
                experiences = self.shared_memory.sample()
                self.agents[i].learn(experiences, GAMMA)
        """


    def act(self, states, add_noise=True):
        actions = np.zeros([self.num_agents, self.action_size])
        for i in range(self.num_agents):
            actions[i, :] = self.agents[i].act(states[i], add_noise)
        return actions

    def save_weights(self):
        for i in range(self.num_agents):
            torch.save(self.agents[i].actor_local.state_dict(), 'agent{}_checkpoint_actor.pth'.format(i+1))
            torch.save(self.agents[i].critic_local.state_dict(), 'agent{}_checkpoint_critic.pth'.format(i+1))

    def reset(self):
        for agent in self.agents:
            agent.reset()
