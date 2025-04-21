import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class QAgent:
    def __init__(self, state_size, action_size, learning_rate=0.0001, gamma=0.95, epsilon=0.5, epsilon_min=0.01, epsilon_decay=0.95, memory_size=3000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.memory_index = 0
        self.memory_full = False
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize memory tensors on GPU
        self.state_memory = torch.zeros((memory_size, state_size), device=self.device)
        self.action_memory = torch.zeros(memory_size, dtype=torch.long, device=self.device)
        self.reward_memory = torch.zeros(memory_size, device=self.device)
        self.next_state_memory = torch.zeros((memory_size, state_size), device=self.device)
        self.done_memory = torch.zeros(memory_size, device=self.device)
        
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize PyTorch model and optimizer
        self.model = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def remember(self, state, action, reward, next_state, done):
        # Convert inputs to tensors and move to GPU
        state = torch.tensor(state, dtype=torch.float32, device=self.device).flatten()
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).flatten()
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)
        
        # Store experience in memory
        idx = self.memory_index % self.memory_size
        
        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.next_state_memory[idx] = next_state
        self.done_memory[idx] = done
        
        self.memory_index += 1
        if self.memory_index >= self.memory_size:
            self.memory_full = True
            self.memory_index = 0
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        self.model.eval()
        with torch.no_grad():
            # Convert state to tensor and move to GPU
            state = torch.tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
            act_values = self.model(state)
        return torch.argmax(act_values).item()
    
    def replay(self, batch_size):
        # Get current memory size
        current_memory_size = self.memory_size if self.memory_full else self.memory_index
        if current_memory_size < batch_size:
            return
            
        # Sample indices
        indices = torch.randint(0, current_memory_size, (batch_size,), device=self.device)
        
        # Get batch from memory tensors (already on GPU)
        states = self.state_memory[indices]
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices]
        next_states = self.next_state_memory[indices]
        dones = self.done_memory[indices]
        
        # Get current Q values for the actions taken
        self.model.train()
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values and compute target Q values
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        
    def save(self, name):
        torch.save(self.model.state_dict(), name) 