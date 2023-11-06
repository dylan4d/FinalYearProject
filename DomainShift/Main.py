# module imports
import numpy as np
from itertools import count
import torch
import torch.optim as optim
import torch.nn.functional as F
import optuna
from matplotlib import pyplot as plt

# custom imports
from CustomCartPoleEnvironmentClass import CustomCartPoleEnv
from ReplayMemoryClass import ReplayMemory
from DQNClass import DQN
from PlotFunction import plot_function
from ActionSelection import ActionSelector
from OptimizeModel import Optimizer


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


    memory = ReplayMemory(REPLAY_MEMORY_SIZE)
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observation = len(state)

    policy_net = DQN(n_observation, n_actions).to(device)
    target_net = DQN(n_observation, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

    action_selector = ActionSelector(policy_net, env.action_space.n, device, EPS_START, EPS_END, EPS_DECAY)
    optimizer_instance = Optimizer(policy_net, target_net, optimizer, memory, device, BATCH_SIZE, GAMMA, TAU)

    episode_durations = []
    losses = optimizer_instance.losses
    eps_thresholds = []
    episode_rewards = []

    num_episodes = 2500
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
            (observation, reward, terminated, truncated, info), domain_shift = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            episode_total_reward += reward.item() # accumulate reward

            if not done:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                next_state = None
            
            memory.push(state, action, next_state, reward)
            state = next_state
            optimizer_instance.Optimizer()

            if done:
                episode_durations.append(t + 1)
                break
        
        episode_rewards.append(episode_total_reward)
        
        if len(episode_rewards) >= 100:
            average_reward = np.mean(episode_rewards[-100:])
            if average_reward > PERFORMANCE_THRESHOLD:
                EPS_START = max(EPS_START * (1-EPS_DECAY), EPS_END)

        
        fig, axs = plt.subplots(4,1, figsize=(10,7))
        plot_function(fig, axs, episode_durations, losses, eps_thresholds, episode_rewards, optimization_mode=False)

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



# study organisation
storage_url = "sqlite:///optuna_study.db"
study_name = 'cartpole_study'

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