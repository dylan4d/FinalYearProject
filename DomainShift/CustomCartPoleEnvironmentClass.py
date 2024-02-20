from gym.envs.classic_control import CartPoleEnv
import random

class CustomCartPoleEnv(CartPoleEnv):
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
        self.original_length = self.length  # Save the original length for resetting
        self.length_change_rate = 0.01  # Define how quickly the pole length should change
        self.min_length_change = -0.01
        self.max_length_change = 0.09
        self.original_masscart = self.masscart # Save the original mass
        self.min_masscart_change = 0.1 
        self.max_masscart_change = 1.0
        self.original_force_mag = self.force_mag # Save the original force
        self.min_force_change = 8.0
        self.max_force_change = 12.0


    def change_pole_length(self):
        length_change = random.uniform(self.min_length_change, self.max_length_change)
        self.length += length_change
        self.polemass_length = (self.masspole * self.length)
        self.total_mass = (self.masspole + self.masscart)

    def change_cart_mass(self):
        masscart_change = random.uniform(self.min_masscart_change, self.max_masscart_change)
        self.masscart += masscart_change
        self.total_mass = self.masscart + self.masspole

    def change_cart_friction(self):
        force_change = random.uniform(self.min_force_change, self.max_force_change)
        self.force_mag = force_change

    def step(self, action):
        self.change_cart_mass()  # Change the pole length at each step
        domain_shift = self.quantify_domain_shift()
        observation, reward, terminated, truncated, info = super().step(action)
        return (observation, reward, terminated, truncated, info), domain_shift

    def reset(self):
        self.length = self.original_length  # Reset the pole length when the environment is reset
        return super().reset()
    
    def quantify_domain_shift(self):
        return abs(self.original_masscart - self.masscart)
    
    def set_logger(self, logger):
        self.logger = logger
    