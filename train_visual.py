import numpy as np
from classes.visual_q_agent import VisualQAgent
from classes.visual_training_env import VisualTrainingEnvironment
import os
import torch

def train_visual(num_episodes=1200, batch_size=64):
    # Create environment
    env = VisualTrainingEnvironment()
    
    # Get initial state to determine input shape
    initial_state = env.reset()
    screen_shape = initial_state[0].shape  # Get the shape of the visual input
    #print(screen_shape)
    action_size = 6  # 0=accelerate, 1=brake, 2=left, 3=right, 4=accelerate and left, 5=accelerate and right
    
    # Create agent with visual input shape
    agent = VisualQAgent(input_shape=screen_shape, action_size=action_size)
    best_reward = 0

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        iteration = 0

        # Collect experiences during the episode
        while not done and iteration < 10000:
            # Get action from agent
            action = agent.act(state)
            # Step environment
            next_state, reward, done = env.step(action)
            # Store experience in memory
            agent.remember(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
            iteration += 1

        if env.gate_reward > best_reward:
            best_reward = env.gate_reward
            os.makedirs("models", exist_ok=True)
            agent.save("models/_best_visual_agent.h5")
        
        batch_count = iteration//batch_size
        batch_losses, batch_mean_qs, batch_std_qs = np.zeros(batch_count), np.zeros(batch_count), np.zeros(batch_count)
        
        # Train on all experiences from this episode
        if iteration > 0:  # Only train if we have experiences
            for i in range(batch_count):
                loss, mean_q, std_q = agent.replay(batch_size)
                if loss > 0 and (episode + 1) % 5 == 0:  # Only update metrics if replay was successful and it's the episode where we print
                    batch_losses[i] = loss
                    batch_mean_qs[i] = mean_q
                    batch_std_qs[i] = std_q

        # Update epsilon and learning rate
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        agent.update_learning_rate()

        # Save model and print statistics periodically
        if (episode + 1) % 1 == 0:
            # Print episode statistics
            print(f"Episode: {episode + 1}/{num_episodes}")
            print(f"  Reward: {episode_reward:.2f}, Gate Reward: {env.gate_reward:.2f}, Speed Reward: {env.speed_reward:.2f}")
            print(f"  Average Loss: {np.mean(batch_losses):.4f}, Average Q-value: {np.mean(batch_mean_qs):.4f}, Q-value Std Dev: {np.mean(batch_std_qs):.4f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Learning Rate: {agent.learning_rate:.6f}")
            print(f"  Iterations: {iteration}")
            print("-------------------")
            
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/visual_agent_episode_{episode + 1}.h5")

    # Close environment
    env.close()

if __name__ == "__main__":
    train_visual() 