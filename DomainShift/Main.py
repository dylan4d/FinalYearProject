# module imports
import numpy as np
from itertools import count
import torch
import torch.nn.functional as F
import optuna
from matplotlib import pyplot as plt

# custom imports
from ReplayMemoryClass import ReplayMemory
from DQNClass import DQN
from PlotFunction import plot_function
from InitEnvironment import config, initialize_environment
from DataLoggerClass import DataLogger


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

    # Use the hyperparameters from the config dictionary
    PERFORMANCE_THRESHOLD = config['performance_threshold']

    # initialise environment and components
    env, policy_net, target_net, optimizer, action_selector, optimizer_instance = initialize_environment(config)
    memory = ReplayMemory(config['replay_memory_size'])
    optimizer_instance.memory = memory
    
    # For plotting function
    fig, axs = plt.subplots(4, 1, figsize=(10, 7))  # Create them once here
    episode_durations = []
    losses = optimizer_instance.losses
    eps_thresholds = []
    episode_rewards = []

    # Logging function
    logger = DataLogger('training_data.csv')
    env.set_logger(logger)

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
            optimizer_instance.optimize()
            cumulative_reward = reward.item() # accumulate reward

            # Log step data
            logger.log_step(
            episode=i_episode,
            step=t,
            original_length=env.original_length,
            current_length=env.length,
            action=action.item(),
            reward=reward.item(),
            domain_shift=domain_shift,
            cumulative_reward=cumulative_reward,
            epsilon=action_selector.get_current_epsilon_threshold(),
            loss=optimizer_instance.get_last_loss()  # You would need to implement this method
            )

            if done:
                episode_durations.append(t + 1)
                break
        
        episode_rewards.append(episode_total_reward)
        
        if len(episode_rewards) >= 100:
            average_reward = np.mean(episode_rewards[-100:])
            if average_reward > PERFORMANCE_THRESHOLD:
                action_selector.update_epsilon()
        
        # Get the current epsilon threshold after update
        current_eps_threshold = action_selector.get_current_epsilon_threshold()
        eps_thresholds.append(current_eps_threshold)  # Append the latest epsilon value

        # Plot the graphs wanted
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