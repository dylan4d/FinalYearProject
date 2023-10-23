import gymnasium as gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import optuna

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.init_weights()
    
    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.bn2(self.layer2(x)))
        return self.layer3(x)
    
    def init_weights(self):
        for layer in [self.layer1, self.layer2, self.layer3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        print('Using CUDA')

set_seed(1)
# best model
global best_value
best_value = -float('inf')

def objective(trial):
    env = gym.make('CartPole-v1')
    global best_value
    # Define the hyperparameter search space
    LR = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    GAMMA = trial.suggest_float('gamma', 0.95, 0.999)
    TARGET_UPDATE = trial.suggest_int('target_update', 5, 20)
    EPS_START = trial.suggest_float('eps_start', 0.8, 1)
    EPS_END = trial.suggest_float('eps_end', 0.01, 0.1)
    EPS_DECAY = trial.suggest_int('eps_decay', 200, 1000)
    BATCH_SIZE = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    REPLAY_MEMORY_SIZE = trial.suggest_categorical('replay_memory_size', [1000, 2000, 3000, 4000, 5000])
    
    memory = ReplayMemory(REPLAY_MEMORY_SIZE)
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observation = len(state)

    policy_net = DQN(n_observation, n_actions).to(device)
    target_net = DQN(n_observation, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    scheduler = scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=10, verbose=True)

    steps_done = 0

    def select_action(state):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        
        # Set the model to evaluation mode
        policy_net.eval()
        
        with torch.no_grad():
            if sample > eps_threshold:
                return policy_net(state).max(1)[1].view(1, 1)
            else:
                return torch.tensor([[env.action_space.sample()]], dtype=torch.long, device=device)

    episode_durations = []

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
            next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_state_actions).squeeze(1)
        
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()

        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        optimizer.step()
        scheduler.step(np.mean(episode_durations))

    # Function to plot the duration of each episode
    def plot_duration(episode_durations, optimization_mode=False):
        if optimization_mode:
            return
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)  # pause a bit so that plots are updated

    num_episodes = 2500
    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Set the policy net to training mode at the start of each episode
        policy_net.train()
        
        for t in count():
            # Select action
            policy_net.eval()
            action = select_action(state)
            policy_net.train()

            # Take action and observe new state
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if not done:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                next_state = None
            
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model()

            if done:
                episode_durations.append(t + 1)
                break

        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        plot_duration(episode_durations, optimization_mode=True)

        trial.report(episode_durations[-1], i_episode)

        if trial.should_prune():
            raise optuna.TrialPruned()
        
        if np.mean(episode_durations) > best_value:
            best_value = np.mean(episode_durations)
            torch.save(policy_net.state_dict(), 'cartpole_v1_best_model.pth')

    return np.mean(episode_durations)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=300)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("Value: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# torch.save(policy_net.state_dict(), '/home/df21/Documents/FYP/cartpole_v1_model.pth')