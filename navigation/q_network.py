import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, neurons=50, seed=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed

        NOTE: skeleton provided by Udacity
        """
        super(QNetwork, self).__init__()
        if seed == 0 or seed:
            self.seed = torch.manual_seed(seed)
        else:
            print('Seed is set to default None.')
        self.l_in = nn.Linear(state_size, neurons)
        self.hidden = nn.Linear(neurons, neurons)
        self.l_out = nn.Linear(neurons, action_size)
        self.activ = nn.ReLU()

    def forward(self, state):
        """Build a network that maps state -> action values."""
        state = self.l_in(state)
        state = self.activ(state)
        state = self.hidden(state)
        state = self.activ(state)
        state = self.l_out(state)
        return state
