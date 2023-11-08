import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DomainShiftPredictor:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, suitability_threshold, adjustment_factor, device):
        self.model = DomainShiftPredictor(input_dim, hidden_dim, output_dim).to(device)
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