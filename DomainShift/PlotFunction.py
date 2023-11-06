import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_function(episode_durations, losses, eps_thresholds, episode_rewards, optimization_mode=False):
        if optimization_mode:
            return
        # Initialize the figure and axes for the plots outside of the function
        fig, axs = plt.subplots(4, 1, figsize=(10, 7))

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