import random
from collections import deque
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Deep Q-Network multi-layer perceptron architecture."""

    def __init__(self, state_dim: int = 4, action_dim: int = 2):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """
    Deep Q-Learning Agent handling experience replay, epsilon-greedy action selection,
    Q-target updates, and model saving/loading.
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.99,
        buffer_capacity: int = 2000,
    ):
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.epsilon_min: float = epsilon_min
        self.epsilon_decay: float = epsilon_decay

        self.memory = deque(maxlen=buffer_capacity)

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.update_target_network()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def update_target_network(self) -> None:
        """Copies Q-network weights to target network for target stabilization."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Stores transition sample in experience replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Selects discrete action using epsilon-greedy exploration/exploitation.
        """
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def replay(self, batch_size: int = 32) -> Optional[float]:
        """
        Trains Q-network on a random minibatch sampled from replay memory.
        Returns loss value as float.
        """
        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(dones).unsqueeze(1)

        # Current Q-values predicted by Q-network
        current_q = self.q_network(states_t).gather(1, actions_t)

        # Target Q-values using Target Network
        with torch.no_grad():
            max_next_q = self.target_network(next_states_t).max(1, keepdim=True)[0]
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(loss.item())

    def save(self, filepath: str) -> None:
        """Saves Q-network weights to file."""
        torch.save(self.q_network.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        """Loads Q-network weights from file."""
        weights = torch.load(filepath)
        self.q_network.load_state_dict(weights)
        self.update_target_network()
