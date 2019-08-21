import torch
import torch.nn as nn

import torch.optim as optim
import numpy as np

import random

from navigation.constants import BUFFER_SIZE, UPDATE_EVERY, BATCH_SIZE, GAMMA, TAU, LR
from navigation.q_network import QNetwork
from navigation.replay_buffer import ReplayBuffer
from navigation.utils import device


class Agent:
    def __init__(self, state_size, action_size, neurons=50, seed=None):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.qnetwork_local = QNetwork(state_size, action_size, neurons, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, neurons, seed).to(device)
        self.qnetwork_target.eval()
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.loss = torch.nn.MSELoss()
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

    def update(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """
        Returns an action given the state.

            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection,
                            if eps is too high, the chances the policy choses an option a random choice are higher, useful for exploration
                            if eps is too low, will use the neural network to get the actions
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.

            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
        with torch.no_grad():
            maxQ_target = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        targets = rewards + gamma * maxQ_target * (1 - dones)
        y = self.qnetwork_local(states)
        y = y.gather(1, actions)
        loss = self.loss(y, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
