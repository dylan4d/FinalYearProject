import torch
import torch.nn.functional as F
from ReplayMemoryClass import Transition

class Optimizer:
    
    def __init__(self, policy_net, target_net, optimizer, replay_memory, device, batch_size, gamma, tau):
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.memory = replay_memory
        self.device = device
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = tau
        self.losses = []

    def Optimizer(self):
            if len(self.memory) < self.BATCH_SIZE:
                return
            transitions = self.memory.sample(self.BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=self.device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
            with torch.no_grad():
                next_state_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_state_actions).squeeze(1)
            
            expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
            self.losses.append(loss.item())  # Store the loss value

            self.optimizer.zero_grad()  # Zero the gradients before the backward pass
            loss.backward()  # Compute the backward pass
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)  # Gradient clipping
            self.optimizer.step()  # Take a step with the optimizer

            # Soft update the target network
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.TAU * policy_param.data + (1.0 - self.TAU) * target_param.data)
            