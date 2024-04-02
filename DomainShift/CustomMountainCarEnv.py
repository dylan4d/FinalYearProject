from gym.envs.classic_control import mountain_car
import random

class CustomMountainCarEnv(mountain_car.MountainCarEnv):
    """
    Custom implementation of the CartPole environment with an adjustable pole length.
    The pole length changes over time, simulating a non-stationary environment.
    
    Attributes:
        original_length (float): The original length of the pole.
        length_change_rate (float): The rate at which the pole length changes each step.
    """
    
    def __init__(self):
        super().__init__()
        # Domain shifts
        self.original_force = self.force
        self.min_force_change = -0.1
        self.max_force_change = 0.2

    def change_force(self):
        force_change = random.uniform(self.min_force_change, self.max_force_change)
        self.force += force_change

    def step(self, action):
        self.change_force()
        domain_shift = self.quantify_domain_shift()
        observation, reward, terminated, truncated, info = super().step(action)
        return (observation, reward, terminated, truncated, info), domain_shift

    def reset(self):
        self.force = self.original_force  # Reset the pole length when the environment is reset
        return super().reset()
    
    def quantify_domain_shift(self):
        return abs(self.original_force - self.force)
    
    def set_logger(self, logger):
        self.logger = logger
    