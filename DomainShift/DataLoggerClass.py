import csv
import os

class DataLogger:
    def __init__(self, filename):
        self.filename = filename
        self.fields = ['episode', 'step', 'original_length', 'current_length', 'action', 'reward', 'domain_shift', 'cumulative_reward', 'epsilon', 'loss']
        self.ensure_file()

    def ensure_file(self):
        if not os.path.isfile(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()

    def log_step(self, episode, step, original_length, current_length, action, reward, domain_shift, cumulative_reward, epsilon, loss):
        with open(self.filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow({
                'episode': episode,
                'step': step,
                'original_length': original_length,
                'current_length': current_length,
                'action': action,
                'reward': reward,
                'domain_shift': domain_shift,
                'cumulative_reward': cumulative_reward,
                'epsilon': epsilon,
                'loss': loss
            })