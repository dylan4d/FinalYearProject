import random
import math
import torch

class ActionSelector:
    def __init__(self, policy_net, num_actions, device, EPS_START, EPS_END, EPS_DECAY):
        self.policy_net = policy_net
        self.num_actions = num_actions
        self.device = device
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.steps_done = 0
        self.eps_thresholds = []

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        self.eps_thresholds.append(eps_threshold)
        
        # Set the model to evaluation mode
        self.policy_net.eval()
        
        with torch.no_grad():
            if sample > eps_threshold:
                return self.policy_net(state).max(1)[1].view(1, 1)
            else:
                return torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long, device=self.device)
    
    def get_epsilon_thresholds(self):
        return self.eps_thresholds
    
    def update_epsilon(self):
        self.EPS_START = max(self.EPS_START * (1 - self.EPS_DECAY), self.EPS_END)