import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainShiftPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DomainShiftPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x # Assuming binary classification of suitability