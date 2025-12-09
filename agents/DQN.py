import random
from collections import deque, namedtuple
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import os

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron for Q-value estimation"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """
    Standard DQN Agent with fixed target network

    Features:
    - Experience replay buffer
    - Target network for stable training
    - Epsilon-greedy exploration with decay
    - Gradient clipping
    """

    def __init__(self, obs_dim, n_actions, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=500000):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_count = 0

        self.q_net = MLP(obs_dim, n_actions).to(DEVICE)
        self.target_net = MLP(obs_dim, n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=10_000)
        self.batch_size = 64
        self.update_target_every = 1000

    def act(self, obs, greedy=False):
        """
        Select action using epsilon-greedy policy

        Args:
            obs: Observation (numpy array)
            greedy: If True, always select best action (for evaluation)

        Returns:
            action: Selected action index
        """
        self.step_count += 1
        if not greedy:
            eps = self._get_epsilon()
            if random.random() < eps:
                return random.randrange(self.n_actions)

        obs_t = (torch.from_numpy(obs).float() / 255.0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_vals = self.q_net(obs_t)
        return int(q_vals.argmax(dim=1).item())

    def _get_epsilon(self):
        """Calculate current epsilon value with linear decay"""
        return max(self.epsilon_end,
                   self.epsilon - (1.0 - self.epsilon_end) * self.step_count / self.epsilon_decay)

    def store(self, *args):
        """Store transition in replay buffer"""
        self.replay_buffer.append(Transition(*args))

    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        batch = Transition(*zip(*batch))

        state = torch.from_numpy(np.stack(batch.state)).float().to(DEVICE) / 255.0
        action = torch.tensor(batch.action).long().to(DEVICE)
        reward = torch.tensor(batch.reward).float().to(DEVICE)
        next_state = torch.from_numpy(np.stack(batch.next_state)).float().to(DEVICE) / 255.0
        done = torch.tensor(batch.done).float().to(DEVICE)

        # Current Q-values
        q_values = self.q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)

        # Target Q-values using target network
        with torch.no_grad():
            next_q = self.target_net(next_state).max(1)[0]
            target = reward + self.gamma * next_q * (1 - done)

        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update target network periodically
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save_checkpoint(self, filepath):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'epsilon': self.epsilon,
        }
        torch.save(checkpoint, filepath)
        print(f"DQN checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        if not os.path.exists(filepath):
            print(f"Checkpoint {filepath} not found")
            return False

        checkpoint = torch.load(filepath, map_location=DEVICE)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
        self.epsilon = checkpoint['epsilon']
        print(f"DQN checkpoint loaded from {filepath}")
        return True