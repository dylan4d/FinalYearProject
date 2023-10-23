import gymnasium as gym
import torch
from CartPole_v1 import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v1')
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

# Load the trained model
model = DQN(n_observations, n_actions).to(device)
model.load_state_dict(torch.load('cartpole_v1_model.pth'))
model.eval()

for i_episode in range(10):  # Run 10 episodes for demonstration
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0
    
    while True:
        env.render()  # Render the current state of the environment
        
        # Select an action using the trained model
        with torch.no_grad():
            action = model(state).max(1)[1].view(1, 1)
        
        # Take the selected action and observe the next state and reward
        next_state, reward, done, _ = env.step(action.item())
        total_reward += reward
        state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
        
        if done:
            print(f'Episode {i_episode + 1}: Total Reward = {total_reward}')
            break

env.close()
