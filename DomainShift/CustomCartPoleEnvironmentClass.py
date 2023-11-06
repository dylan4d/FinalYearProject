from gym.envs.classic_control import CartPoleEnv

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
        self.original_length = self.length  # Save the original length for resetting
        self.length_change_rate = 0.01  # Define how quickly the pole length should change

    def change_pole_length(self):
        self.length += self.length_change_rate
        self.polemass_length = (self.masspole * self.length)
        self.total_mass = (self.masspole + self.masscart)

    def step(self, action):
        self.change_pole_length()  # Change the pole length at each step
        domain_shift = self.quantify_domain_shift()
        return super().step(action), domain_shift

    def reset(self):
        self.length = self.original_length  # Reset the pole length when the environment is reset
        return super().reset()
    
    def quantify_domain_shift(self):
        return abs(self.original_length - self.length)
    