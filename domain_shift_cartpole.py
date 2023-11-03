import gym
from gym.envs.classic_control import CartPoleEnv
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

class CustomCartPoleEnv(CartPoleEnv):
    def __init__(self):
        super().__init__()
        self.original_length = self.length  # Save the original length for resetting
        self.length_change_rate = 0.01  # Define how quickly the pole length should change

    def change_pole_length(self):
        self.length += self.length_change_rate
        self.polemass_length = (self.masspole * self.length)
        self.total_mass = (self.masspole + self.masscart)
        self.polemass_length = (self.masspole * self.length)

    def step(self, action):
        self.change_pole_length()  # Change the pole length at each step
        return super().step(action)

    def reset(self):
        self.length = self.original_length  # Reset the pole length when the environment is reset
        return super().reset()

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
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, n_actions)
        self.init_weights()
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
    
    def init_weights(self):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
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
    env = CustomCartPoleEnv()
    global best_value
    # Define the hyperparameter search space
    LR = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    GAMMA = trial.suggest_float('gamma', 0.95, 0.999)
    TARGET_UPDATE = trial.suggest_int('target_update', 500, 2000)
    EPS_START = trial.suggest_float('eps_start', 0.8, 1)
    EPS_END = trial.suggest_float('eps_end', 0.05, 0.05)
    EPS_DECAY = trial.suggest_int('eps_decay', 2000, 10000)
    BATCH_SIZE = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    REPLAY_MEMORY_SIZE = trial.suggest_categorical('replay_memory_size', [1000, 2000, 3000, 4000, 5000])
    PERFORMANCE_THRESHOLD = 195


    memory = ReplayMemory(REPLAY_MEMORY_SIZE)
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observation = len(state)

    policy_net = DQN(n_observation, n_actions).to(device)
    target_net = DQN(n_observation, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    steps_done = 0

    def select_action(state):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        eps_thresholds.append(eps_threshold)  # Store the epsilon value
        
        # Set the model to evaluation mode
        policy_net.eval()
        
        with torch.no_grad():
            if sample > eps_threshold:
                return policy_net(state).max(1)[1].view(1, 1)
            else:
                return torch.tensor([[env.action_space.sample()]], dtype=torch.long, device=device)

    episode_durations = []
    losses = []
    eps_thresholds = []
    episode_rewards = []

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
        losses.append(loss.item())  # Store the loss value

        optimizer.zero_grad()
        loss.backward()

        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        optimizer.step()

    # Initialize the figure and axes for the plots outside of the function
    fig, axs = plt.subplots(4, 1, figsize=(10, 7))

    def plot_durations(episode_durations, losses, eps_thresholds, episode_rewards, optimization_mode=False):
        if optimization_mode:
            return

        # Clear the current axes and figure
        for ax in axs:
            ax.cla()

        # Plot Episode Durations
        axs[0].set_title('Training Metrics')
        axs[0].set_ylabel('Duration')
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        axs[0].plot(durations_t.numpy(), label='Raw')
        smoothed_durations = np.cumsum(durations_t.numpy()) / (np.arange(len(durations_t)) + 1)
        axs[0].plot(smoothed_durations, color='orange', label='Smoothed')
        axs[0].legend()

        # Plot Losses
        axs[1].set_ylabel('Loss')
        losses_t = np.array(losses)
        axs[1].plot(losses_t, label='Raw')
        smoothed_losses = np.cumsum(losses_t) / (np.arange(len(losses_t)) + 1)
        axs[1].plot(smoothed_losses, color='orange', label='Smoothed')
        axs[1].legend()

        # Plot Epsilon Thresholds
        axs[2].set_ylabel('Epsilon')
        axs[2].plot(eps_thresholds, label='Epsilon')
        axs[2].legend()

        # Plot Reward
        axs[3].set_xlabel('Episode')
        axs[3].set_ylabel('Reward')
        rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
        axs[3].plot(rewards_t.numpy(), label='Episode Reward')
        axs[3].legend()

        plt.tight_layout()
        plt.pause(0.001)  # pause a bit so that plots are updated

    num_episodes = 25000
    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_total_reward = 0 # reset the total reward for the episode
        
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
            episode_total_reward += reward.item() # accumulate reward

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
        
        episode_rewards.append(episode_total_reward)
        
        if len(episode_rewards) >= 115:
            average_reward = np.mean(episode_rewards[-100:])
            if average_reward > PERFORMANCE_THRESHOLD:
                EPS_START = max(EPS_START * EPS_DECAY, EPS_END)

        plot_durations(episode_durations, losses, eps_thresholds, episode_rewards, optimization_mode=False)

        trial.report(episode_durations[-1], i_episode)

        # if trial.should_prune():
        #     raise optuna.TrialPruned()
        
        if np.mean(episode_durations) > best_value:
            best_value = np.mean(episode_durations)
            torch.save(policy_net.state_dict(), 'cartpole_v1_best_model.pth')

    return np.mean(episode_durations)

storage_url = "sqlite:///optuna_study.db"
study_name = 'cartpole_study'

# Create a new study or load an existing study
study = optuna.create_study(study_name=study_name, storage=storage_url, direction='maximize', load_if_exists=True)
study.optimize(objective, n_trials=500)

# Load the study
study = optuna.load_study(study_name=study_name, storage=storage_url)
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("Value: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# torch.save(policy_net.state_dict(), '/home/df21/Documents/FYP/cartpole_v1_model.pth')