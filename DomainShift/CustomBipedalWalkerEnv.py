from gym.envs.box2d import bipedal_walker
import random

class CustomBipedalWalkerEnv(bipedal_walker.BipedalWalker):
    def __init__(self):
        super().__init__()
        # Domain shifts
        self.original_gravity = self.world.gravity
        self.min_gravity_change = -1.0
        self.max_gravity_change = 2.0
    
    def change_gravity(self):
        gravity_change = random.uniform(self.min_gravity_change, self.max_gravity_change)
        self.world.gravity = (0.0, self.original_gravity[1] + gravity_change)

    def step(self, action):
        self.change_gravity()
        domain_shift = self.quantify_domain_shift()
        observation, reward, terminated, truncated, info = super().step(action)
        self.logger.log_step(self.episode, self.step, self.original_gravity[1], self.world.gravity[1], action, reward, domain_shift)
        return (observation, reward, terminated, truncated, info), domain_shift

    def reset(self):
        self.world.gravity = self.original_gravity  # Reset the gravity when the environment is reset
        return super().reset()
    
    def quantify_domain_shift(self):
        return abs(self.original_gravity[1] - self.world.gravity[1])
    
    def set_logger(self, logger):
        self.logger = logger