import numpy as np
import torch.nn as nn
import torch
from torch import optim
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOPolicy(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, n_actions)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        feat = self.shared(x)
        logits = self.policy_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return logits, value


class PPOAgent:
    def __init__(self, obs_dim, n_actions, lr=1e-4, gamma=0.99, lam=0.95, clip_eps=0.2):
        self.n_actions = n_actions
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps

        self.net = PPOPolicy(obs_dim, n_actions).to(DEVICE)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.trajectory = []  # store (s, a, logp, r, v, done)

    def act(self, obs, greedy=False):
        obs_t = (torch.from_numpy(obs).float() / 255.0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, value = self.net(obs_t)
        if greedy:
            # deterministic (for evaluation)
            action = torch.argmax(logits, dim=1)
            return int(action.item()), 0.0, float(value.item())
        else:
            # stochastic (for training)
            probs = torch.distributions.Categorical(logits=logits)
            action = probs.sample()
            logp = probs.log_prob(action)

        return int(action.item()), float(logp.item()), float(value.item())

    def store(self, transition):
        # transition: (state, action, logp, reward, value, done)
        self.trajectory.append(transition)

    def finish_trajectory_and_update(self):
        if not self.trajectory:
            return

        # unpack
        states, actions, logps, rewards, values, dones = zip(*self.trajectory)
        self.trajectory = []

        states = torch.from_numpy(np.stack(states)).float().to(DEVICE) / 255.0
        actions = torch.tensor(actions).long().to(DEVICE)
        old_logps = torch.tensor(logps).float().to(DEVICE)
        values = torch.tensor(values).float().to(DEVICE)
        rewards = torch.tensor(rewards).float().to(DEVICE)
        dones = torch.tensor(dones).float().to(DEVICE)

        # compute advantages with GAE
        advantages = []
        gae = 0.0
        returns = []
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            next_value = values[t]
            returns.insert(0, gae + values[t])

        advantages = torch.tensor(advantages).float().to(DEVICE)
        returns = torch.tensor(returns).float().to(DEVICE)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # multiple epochs over same data (small for demo)
        for _ in range(8):
            logits, value_pred = self.net(states)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(actions)
            ratio = torch.exp(logp - old_logps)

            # PPO clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(value_pred, returns)
            entropy_loss = -dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss + 0.05 * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
            self.optimizer.step()

    def save_checkpoint(self, filepath):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"PPO checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        if not os.path.exists(filepath):
            print(f"Checkpoint {filepath} not found")
            return False

        checkpoint = torch.load(filepath, map_location=DEVICE)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"PPO checkpoint loaded from {filepath}")
        return True