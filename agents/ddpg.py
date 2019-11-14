from agents.actorCritic import Actor, Critic
from agents.ounoiseBuffer import OUNoise, Buffer
import numpy as np

class DDPG():
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        self.critic_target.model.set_weights(self.critic_local.model.get_weights()) 
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        self.phi = 0
        self.theta = .075
        self.psi = .3
        self.noise = OUNoise(self.action_size, self.phi, self.theta, self.psi)

        # Replay memory
        self.buffer_size = 100000
        self.batch = 64
        self.memory = Buffer(self.buffer_size, self.batch)

        self.discount = 0.65  # discount factor 
        self.softUpdate = 0.02  

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
        self.memory.add(self.last_state, action, reward, next_state, done)
        if len(self.memory) > self.batch:
            experiences = self.memory.sample()
            self.learn(experiences)
        self.last_state = next_state

    def act(self, states):
        state = np.reshape(states, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())  

    def learn(self, learnings):
        states = np.vstack([l.state for l in learnings if l is not None])
        actions = np.array([l.action for l in learnings if l is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([l.reward for l in learnings if l is not None]).astype(np.float32).reshape(-1, 1)
        doneArray = np.array([l.done for l in learnings if l is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([l.next_state for l in learnings if l is not None])

        actions_next = self.actor_target.model.predict_on_batch(next_states)
        qTargets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])
        qTargets = rewards + self.discount * qTargets_next * (1 - doneArray)
        self.critic_local.model.train_on_batch(x=[states, actions], y=qTargets)

        gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, gradients, 1])  

        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights)

        new_weights = self.softUpdate * local_weights + (1 - self.softUpdate) * target_weights
        target_model.set_weights(new_weights)