import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    A simple Deep Q-Network architecture.
    
    Attributes:
        n_observations (int): The number of observations from the environment.
        n_actions (int): The number of possible actions the agent can take.
    """
    def __init__(self, n_observations, n_actions, domain_shift_input_dim):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations + domain_shift_input_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    
    def forward(self, x, domain_shift):
        domain_shift = domain_shift.view(-1, 1)  # Reshape to [batch_size, 1]
        x = torch.cat((x, domain_shift), dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)