import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from classes.q_agent import QAgent, SumTree

class VisualQNetwork(nn.Module):
    def __init__(self, input_shape, action_size):
        super(VisualQNetwork, self).__init__()
        
        # CNN layers for processing visual input
        self.conv1 = nn.Conv2d(1, 16, kernel_size=9, stride=4)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        
        # Calculate the size of the flattened features
        self.conv_output_size = self._get_conv_output_size(input_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size + 4, 256)  # +4 for speed, vel_x, vel_y, angle
        self.fc2 = nn.Linear(256, action_size)
        
    def _get_conv_output_size(self, shape):
        # Create a dummy input to calculate the output size
        x = torch.zeros(1, 1, *shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(np.prod(x.shape))
        
    def forward(self, visual_input, car_state):
        # Process visual input through CNN
        x = visual_input.unsqueeze(1)  # Add channel dimension
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        
        # Combine with car state
        x = torch.cat([x, car_state], dim=1)
        
        # Process through fully connected layers
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class VisualQAgent():
    def __init__(self, input_shape, action_size, learning_rate=0.00025, gamma=0.95, epsilon=0.85,
                 epsilon_min=0.12, epsilon_decay=0.995, memory_size=3000, target_update_frequency=5000,
                 lr_decay=0.995, lr_min=0.001, alpha=0.6, beta=0.4, beta_increment=0.001):
        
        self.state_size = input_shape
        self.action_size = action_size
        self.memory_size = memory_size
        self.target_update_frequency = target_update_frequency
        self.update_counter = 0
        
        # Prioritized Experience Replay parameters
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.memory = SumTree(memory_size)
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Learning rate decay parameters
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.lr_min = lr_min
        
        # Initialize PyTorch model and target model with visual input
        self.model = VisualQNetwork(input_shape, action_size).to(self.device)
        self.target_model = VisualQNetwork(input_shape, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def remember(self, state, action, reward, next_state, done):
        # Convert inputs to tensors
        visual_input = torch.tensor(state[0], dtype=torch.float32, device=self.device)
        car_state = torch.tensor(state[1:], dtype=torch.float32, device=self.device)
        
        next_visual_input = torch.tensor(next_state[0], dtype=torch.float32, device=self.device)
        next_car_state = torch.tensor(next_state[1:], dtype=torch.float32, device=self.device)
        
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)
        
        # Store experience with maximum priority
        experience = ((visual_input, car_state), action, reward, (next_visual_input, next_car_state), done)
        self.memory.add(self.max_priority, experience)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        self.model.eval()
        with torch.no_grad():
            visual_input = torch.tensor(state[0], dtype=torch.float32, device=self.device).unsqueeze(0)
            car_state = torch.tensor(state[1:], dtype=torch.float32, device=self.device).unsqueeze(0)
            act_values = self.model(visual_input, car_state)
        return torch.argmax(act_values).item()
    
    def replay(self, batch_size):
        if self.memory.n_entries < batch_size:
            return 0, 0, 0
            
        # Calculate importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        total_priority = self.memory.total()
        segment = total_priority / batch_size
        
        # Sample experiences based on priorities
        indices = []
        priorities = []
        experiences = []
        weights = []
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, experience = self.memory.get(s)
            indices.append(idx)
            priorities.append(priority)
            experiences.append(experience)
            weights.append((priority / total_priority) ** (-self.beta))
            
        # Normalize weights
        weights = torch.tensor(weights, device=self.device)
        weights = weights / weights.max()
        
        # Prepare batch
        visual_inputs = torch.stack([exp[0][0] for exp in experiences])
        car_states = torch.stack([exp[0][1] for exp in experiences])
        actions = torch.stack([exp[1] for exp in experiences])
        rewards = torch.stack([exp[2] for exp in experiences])
        next_visual_inputs = torch.stack([exp[3][0] for exp in experiences])
        next_car_states = torch.stack([exp[3][1] for exp in experiences])
        dones = torch.stack([exp[4] for exp in experiences])
        
        # Get current Q values
        self.model.train()
        current_q_values = self.model(visual_inputs, car_states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values and compute target Q values
        with torch.no_grad():
            next_q_values = self.target_model(next_visual_inputs, next_car_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        # Compute TD errors and update priorities
        td_errors = torch.abs(target_q_values - current_q_values.squeeze()).detach().cpu().numpy()
        for idx, td_error in zip(indices, td_errors):
            priority = (td_error + 1e-6) ** self.alpha
            self.memory.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
            
        # Compute weighted loss
        loss = (weights * nn.MSELoss(reduction='none')(current_q_values.squeeze(), target_q_values)).mean()
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            
        # Calculate Q-value statistics
        with torch.no_grad():
            q_values = self.model(visual_inputs, car_states)
            mean_q = q_values.mean().item()
            std_q = q_values.std().item()
            
        return loss.item(), mean_q, std_q 
    
    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        
    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def update_learning_rate(self):
        """Update the learning rate with decay"""
        if self.learning_rate > self.lr_min:
            self.learning_rate *= self.lr_decay
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
