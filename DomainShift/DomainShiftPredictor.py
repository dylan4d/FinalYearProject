import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DomainShiftPredictor:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, suitability_threshold, adjustment_factor, device):
        self.model = DomainShiftNN(input_dim, hidden_dim, output_dim).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, amsgrad=True)
        self.loss_fn = nn.BCELoss()
        self.suitability_threshold = suitability_threshold
        self.adjustment_factor = adjustment_factor
        self.device = device

    def predict_suitability(self, state, domain_shift_metric):
        with torch.no_grad():  # prediction does not require gradient
            predictor_input = torch.cat((state.flatten(), domain_shift_metric.unsqueeze(0)), dim=0)
            predicted_suitability = self.model(predictor_input.unsqueeze(0))
        return predicted_suitability

    def update(self, state, domain_shift_metric, true_suitability):
        predicted_suitability = self.predict_suitability(state, domain_shift_metric)
        self.optimizer.zero_grad()
        loss = self.loss_fn(predicted_suitability, true_suitability)
        loss.backward()
        self.optimizer.step()
        return loss, predicted_suitability

class DomainShiftNN(nn.Module):
    """
    Neural Network for assessing the domain shift suitability.
    
    Attributes:
        fc1 (torch.nn.Linear): First fully connected layer.
        fc2 (torch.nn.Linear): Second fully connected layer.
        fc3 (torch.nn.Linear): Third fully connected layer, output layer.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initializes the neural network layers.

        Args:
            input_dim (int): The dimension of the input (state + domain shift value).
            hidden_dim (int): The dimension of the hidden layers.
            output_dim (int): The dimension of the output layer.
        """
        super(DomainShiftNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output of the network.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Assuming binary classification (0 or 1) for suitability
        return x