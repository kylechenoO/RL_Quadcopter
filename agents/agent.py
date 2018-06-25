import numpy as np
from agents.model import Actor, Critic
from agents.noise import Noise
from agents.replay import Replay

class Agent(object):
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        self.MU = 0
        self.THETA = 0.15
        self.SIGMA = 0.10
        self.GAMMA = 0.99
        self.TAU = 0.001
        self.BATCHS = 256
        self.MAX_REWARD = -999999999

        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())

        self.noiseObj = Noise(self.action_size, self.MU, self.THETA, self.SIGMA)
        self.replayObj = Replay(self.BATCHS)

    def reset_episode(self):
        self.count = 0
        self.total_reward = 0
        self.noiseObj.reset()
        state = self.task.reset()
        self.last_state = state
        return(state)

    def step(self, action, reward, next_state, done):
        self.replayObj.add(self.last_state, action, reward, next_state, done)
        self.total_reward += reward
        self.count += 1
        if self.total_reward > self.MAX_REWARD:
            self.MAX_REWARD = self.total_reward

        if len(self.replayObj) > self.BATCHS:
            experiences = self.replayObj.sample()
            self.learn(experiences)

        self.last_state = next_state

    def act(self, states):
        action = self.actor_local.model.predict(np.reshape(states, [-1, self.state_size]))[0]
        return(list(action + self.noiseObj.sample()))

    def learn(self, experiences):
        states = np.array([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).reshape(-1, 1)

        next_sta = np.array([e.next_state for e in experiences if e is not None])
        next_act = self.actor_target.model.predict_on_batch(next_sta)
        next_tgt= self.critic_target.model.predict_on_batch([next_sta, next_act])
        tgt = rewards + self.GAMMA * next_tgt * (1 - dones)

        self.critic_local.model.train_on_batch(x=[states, actions], y=tgt)
        gd = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))

        self.actor_local.train_fn([states, gd, 1])
        self.update(self.critic_local.model, self.critic_target.model)
        self.update(self.actor_local.model, self.actor_target.model)   

    def update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.TAU * local_weights + (1 - self.TAU) * target_weights
        target_model.set_weights(new_weights)
