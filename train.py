import numpy as np
from classes.q_agent import QAgent
from classes.training_env import TrainingEnvironment
import os

def train(num_episodes=500, batch_size=16):
    
    # Create environment
    env = TrainingEnvironment()
    # Get state size from the car's get_state method
    initial_state = env.reset()
    state_size = len(initial_state)
    action_size = 7  #0=no action, 1=accelerate, 2=brake, 3=left, 4=right, 5 = accelerate and left, 6 = accelerate and right
    # Create agent
    agent = QAgent(state_size=state_size, action_size=action_size)
    
    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        iteration = 0

        while not done and iteration < 150000:
            # Get action from agent
            action = agent.act(state)
            # Step environment
            next_state, reward, done = env.step(action)
            # Update agent
            agent.remember(state, action, reward, next_state, done)
            agent.replay(batch_size)
            episode_reward += reward
            state = next_state
            iteration += 1
        # Print episode statistics
        print(f"Episode: {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Iteration: {iteration}")
        
        # Save model periodically
        if (episode + 1) % 5 == 0:
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/agent_episode_{episode + 1}.h5")

    # Close environment
    env.close()


if __name__ == "__main__":
    train() 