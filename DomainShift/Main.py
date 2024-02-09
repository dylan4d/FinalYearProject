# module imports
import numpy as np
from itertools import count
import torch
import torch.nn.functional as F
import optuna
from matplotlib import pyplot as plt
import torch.optim as optim
import torch.nn as nn

# custom imports
from ReplayMemoryClass import ReplayMemory
from DQNClass import DQN
from PlotFunction import plot_function
from InitEnvironment import config, initialize_environment
from DataLoggerClass import DataLogger
from DomainShiftPredictor import DomainShiftPredictor


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

env, policy_net, target_net, optimizer, action_selector, optimizer_instance = initialize_environment(config)


def objective(trial):
    global best_value

    # suggest values for tunable hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    eps_decay = trial.suggest_int('eps_decay', 100, 1000)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    gamma = trial.suggest_uniform('gamma', 0.8, 0.9999)
    
    # Update the config with the suggested values
    config.update({
        "lr": lr,
        "eps_decay": eps_decay,
        "batch_size": batch_size,
        "gamma": gamma,
    })

    # reinitialize the environment with the updated values
    env, policy_net, target_net, optimizer, action_selector, optimizer_instance = initialize_environment(config)


    # Use the hyperparameters from the config dictionary
    PERFORMANCE_THRESHOLD = config['performance_threshold']

    # initialise environment and components
    memory = ReplayMemory(config['replay_memory_size'])
    optimizer_instance.memory = memory

    # domain shift predictor necessary values
    input_dim = env.observation_space.shape[0] + 1 # size of the input (state + domain shift value)
    hidden_dim = 128 # size of the hidden layers
    output_dim = 1 # Size of the output (1 if suitable, 0 otherwise)
    suitability_threshold = 0.5
    adjustment_factor = 0.9 # factor to readjust hyperparams

    # Instantiate the domain shift class
    domain_shift_module = DomainShiftPredictor(input_dim, hidden_dim, output_dim, lr, suitability_threshold, adjustment_factor, device)            

    # For plotting function
    fig, axs = plt.subplots(4, 1, figsize=(10, 7))  # Create them once here
    episode_durations = []
    losses = optimizer_instance.losses
    eps_thresholds = []
    episode_rewards = []

    # Logging function
    logger = DataLogger('training_data_with_predictor.csv')
    env.set_logger(logger)

    num_episodes = 2500
    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_total_reward = 0 # reset the total reward for the episode

        # Set the policy net to training mode at the start of each episode
        policy_net.train()
        
        for t in count():

            # Calculate domain shift and domain shift tensor
            domain_shift_metric = env.quantify_domain_shift()
            domain_shift_tensor = torch.tensor([domain_shift_metric], dtype=torch.float32, device=device)

            # Select action
            action = action_selector.select_action(state, domain_shift_tensor)

            # Take action and observe new state
            (observation, reward, terminated, truncated, info), domain_shift = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            reward = torch.tensor([reward], device=device)

            true_suitability = torch.tensor([[1.0]], device=device) if not (terminated or truncated) else torch.tensor([[0.0]], device=device)
            loss, predicted_suitability = domain_shift_module.update(state, domain_shift_tensor, true_suitability)
            
            done = terminated or truncated
            episode_total_reward += reward.item() # accumulate reward

            if not done:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                next_state = None
            
            memory.push(state, action, next_state, reward, domain_shift_tensor)
            state = next_state
            loss = optimizer_instance.optimize()

            if loss is not None:
                # Log step data
                logger.log_step(
                episode=i_episode,
                step=t,
                original_length=env.original_length,
                current_length=env.length,
                action=action.item(),
                reward=reward.item(),
                domain_shift=domain_shift,
                cumulative_reward=episode_total_reward,
                epsilon=action_selector.get_epsilon_thresholds()[-1],
                loss=loss.item()  # This assumes `optimize()` returns a loss, otherwise you'll need to get it another way
                )

            if done:
                episode_durations.append(t + 1)
                break

        if predicted_suitability.item() < suitability_threshold:
            min_epsilon_increase = 0.1  # Define a minimum increase to ensure noticeable changes
            action_selector.EPS_START = min(max(action_selector.EPS_START, action_selector.EPS_END + min_epsilon_increase), 1.0)
        
        episode_rewards.append(episode_total_reward)

        if len(episode_rewards) >= 100:
            average_reward = np.mean(episode_rewards[-100:])
            if average_reward > PERFORMANCE_THRESHOLD:
                action_selector.update_epsilon()
        
        # Get the current epsilon threshold after update
        current_eps_threshold = action_selector.get_epsilon_thresholds()[-1]
        eps_thresholds.append(current_eps_threshold)  # Append the latest epsilon value

        # Plot the graphs wanted
        plot_function(fig, axs, episode_durations, losses, eps_thresholds, episode_rewards, optimization_mode=False)

        trial.report(episode_durations[-1], i_episode)

        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    if mean_reward > best_value:
        best_value = mean_reward
        torch.save(policy_net.state_dict(), 'cartpole_v1_best_model_DSP.pth')

    return mean_reward

# study organisation
storage_url = "sqlite:///optuna_study.db"
study_name = 'cartpole_study_DSP'

# Create a new study or load an existing study
pruner = optuna.pruners.PercentilePruner(99)
study = optuna.create_study(study_name=study_name, storage=storage_url, direction='maximize', load_if_exists=True, pruner=pruner)


try:
    study.optimize(objective, n_trials=100)
except Exception as e:
    print(f"An error occurred during optimization: {e}")


# After optimization, use the best trial to set the state of policy_net
best_trial = study.best_trial
best_model_path = 'cartpole_v1_best_model_DSP.pth'
best_model_state = torch.load(best_model_path)

# Reinitialize the environment with the best trial's hyperparameters
config.update(best_trial.params)
env, policy_net, target_net, optimizer, action_selector, optimizer_instance = initialize_environment(config)

policy_net.load_state_dict(best_model_state)
torch.save(policy_net.state_dict(), 'cartpole_v1_best_model_DSP.pth')

# Load the study
study = optuna.load_study(study_name=study_name, storage=storage_url)
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("Value: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")