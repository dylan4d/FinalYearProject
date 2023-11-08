import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainShiftPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DomainShiftPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # Assuming binary classification of suitability

# Example usage:
# Assume `state` is the current state, `domain_shift_metric` is the current domain shift,
# and `policy_params` are the current policy parameters.
# You would concatenate these into a single tensor to serve as input to the model.
predictor = DomainShiftPredictor(input_dim, hidden_dim, output_dim)
predicted_performance = predictor(torch.cat([state, domain_shift_metric, policy_params], dim=1))
