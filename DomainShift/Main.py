# module imports
import math
import random
import numpy as np
from itertools import count
import torch
import torch.optim as optim
import torch.nn.functional as F
import optuna

# custom imports
from DomainShift.CustomCartPoleEnvironmentClass import CustomCartPoleEnv
from ReplayMemoryClass import ReplayMemory, Transition
from DQNClass import DQN
from PlotFunction import plot_function
from ActionSelection import ActionSelector


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    LR = 1e-4
    GAMMA = 0.99
    TAU = 0.005  # For soft update of target parameters
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    BATCH_SIZE = 128
    REPLAY_MEMORY_SIZE = 10000
    PERFORMANCE_THRESHOLD = 195

    action_selector = ActionSelector(policy_net, env.action_space.n, device, EPS_START, EPS_END, EPS_DECAY)

    memory = ReplayMemory(REPLAY_MEMORY_SIZE)
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observation = len(state)

    policy_net = DQN(n_observation, n_actions).to(device)
    target_net = DQN(n_observation, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    steps_done = 0

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

        optimizer.zero_grad()  # Zero the gradients before the backward pass
        loss.backward()  # Compute the backward pass
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)  # Gradient clipping
        optimizer.step()  # Take a step with the optimizer

        # Soft update the target network
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)
        
    
    num_episodes = 25000
    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_total_reward = 0 # reset the total reward for the episode
        
        # Set the policy net to training mode at the start of each episode
        policy_net.train()
        
        for t in count():
            # Select action
            action = action_selector.select_action(state)

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
        
        episode_rewards.append(episode_total_reward)
        
        if len(episode_rewards) >= 100:
            average_reward = np.mean(episode_rewards[-100:])
            if average_reward > PERFORMANCE_THRESHOLD:
                EPS_START = max(EPS_START * (1-EPS_DECAY), EPS_END)

        plot_function(episode_durations, losses, eps_thresholds, episode_rewards, optimization_mode=False)

        trial.report(episode_durations[-1], i_episode)

        if trial.should_prune():
            raise optuna.TrialPruned()
        
        if np.mean(episode_durations) > best_value:
            best_value = np.mean(episode_durations)
            torch.save(policy_net.state_dict(), 'cartpole_v1_best_model.pth')

    # At the end of the objective function, replace the return statement with:
    if len(episode_rewards) >= 100:
        # Return the mean of the last 100 episode rewards
        return np.mean(episode_rewards[-100:])
    else:
        # Not enough episodes to compute the last 100 episodes' mean, return the mean of what we have
        return np.mean(episode_rewards)

storage_url = "sqlite:///optuna_study.db"
study_name = 'cartpole_study'

optuna.delete_study(study_name=study_name, storage=storage_url)

# Create a new study or load an existing study
pruner = optuna.pruners.PercentilePruner(5)
study = optuna.create_study(study_name=study_name, storage=storage_url, direction='maximize', load_if_exists=True, pruner=pruner)
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