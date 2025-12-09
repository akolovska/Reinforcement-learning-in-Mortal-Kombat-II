import random
from collections import deque, namedtuple
import numpy as np
import torch.nn as nn
import torch
from torch import optim
import os

from helpers.PERReplayBuffer import PERReplayBuffer

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DuelingMLP(nn.Module):
    """
    Dueling Network Architecture:
    - Separate streams for Value V(s) and Advantage A(s,a)
    - Combined as: Q(s,a) = V(s) + [A(s,a) - mean(A(s,a))]
    """

    def __init__(self, input_dim, n_actions):
        super().__init__()
        hidden = 512

        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Value stream V(s) - estimates state value
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Advantage stream A(s,a) - estimates action advantages
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        f = self.feature(x)
        V = self.value_stream(f)  # shape: (B,1)
        A = self.adv_stream(f)  # shape: (B,n_actions)

        # Combine streams with mean advantage subtraction for identifiability
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q


class DuelingDDQNAgent:
    """
    Dueling Double DQN Agent with optional Prioritized Experience Replay

    Features:
    - Dueling architecture: Separate value and advantage streams
    - Double DQN: Uses online network for action selection, target network for evaluation
    - Optional PER: Prioritizes important transitions for faster learning
    - Epsilon-greedy exploration with decay
    """

    def __init__(self, obs_dim, n_actions, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=500000,
                 use_per=False):

        self.gamma = gamma
        self.n_actions = n_actions
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_count = 0
        self.use_per = use_per

        # Dueling networks
        self.q_net = DuelingMLP(obs_dim, n_actions).to(DEVICE)
        self.target_net = DuelingMLP(obs_dim, n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Use PER or standard replay buffer
        if use_per:
            self.replay = PERReplayBuffer(capacity=10_000, alpha=0.6, beta_start=0.4, beta_frames=100000)
        else:
            self.replay = deque(maxlen=10_000)

        self.batch_size = 64
        self.update_target_every = 1000

    def act(self, obs, greedy=False):
        """Select action using epsilon-greedy policy"""
        self.step_count += 1
        if not greedy:
            # Decay epsilon
            eps = max(self.epsilon_end,
                      self.epsilon - (1.0 - self.epsilon_end) * self.step_count / self.epsilon_decay)
            if random.random() < eps:
                return random.randint(0, self.n_actions - 1)

        # Greedy action selection
        obs_t = (torch.from_numpy(obs).float() / 255.0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_vals = self.q_net(obs_t)
        return int(q_vals.argmax(dim=1).item())

    def store(self, *args):
        """Store transition in replay buffer"""
        transition = Transition(*args)
        if self.use_per:
            self.replay.add(transition)
        else:
            self.replay.append(transition)

    def train_step(self):
        """Perform one training step"""
        if len(self.replay) < self.batch_size:
            return

        # Sample from replay buffer
        if self.use_per:
            batch, indices, weights = self.replay.sample(self.batch_size)
            batch = Transition(*zip(*batch))
            weights = torch.from_numpy(weights).float().to(DEVICE)
        else:
            batch = random.sample(self.replay, self.batch_size)
            batch = Transition(*zip(*batch))
            indices = None
            weights = torch.ones(self.batch_size).to(DEVICE)

        states = torch.from_numpy(np.stack(batch.state)).float().to(DEVICE) / 255.0
        actions = torch.tensor(batch.action).long().to(DEVICE)
        rewards = torch.tensor(batch.reward).float().to(DEVICE)
        next_states = torch.from_numpy(np.stack(batch.next_state)).float().to(DEVICE) / 255.0
        dones = torch.tensor(batch.done).float().to(DEVICE)

        # Q(s,a) from Dueling network
        q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target: use online network for action selection, target network for Q-value
        with torch.no_grad():
            online_next_q = self.q_net(next_states)
            next_actions = online_next_q.argmax(dim=1, keepdim=True)

            target_next_q = self.target_net(next_states)
            next_q = target_next_q.gather(1, next_actions).squeeze(1)

            target = rewards + self.gamma * next_q * (1 - dones)

        # Calculate TD errors for PER priority update
        td_errors = (q - target).detach().cpu().numpy()

        # Weighted loss for importance sampling (PER)
        loss = (weights * nn.functional.mse_loss(q, target, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update priorities if using PER
        if self.use_per and indices is not None:
            self.replay.update_priorities(indices, td_errors)

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
        print(f"Dueling DDQN checkpoint saved to {filepath}")

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
        print(f"Dueling DDQN checkpoint loaded from {filepath}")
        return True