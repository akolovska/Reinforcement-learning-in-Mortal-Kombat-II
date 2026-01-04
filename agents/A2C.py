import os
import torch
import torch.nn as nn
from torch import optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class A2CNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 512),
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


class A2CAgent:
    def __init__(self, obs_dim, n_actions, lr=1e-4, gamma=0.99, entropy_coef=0.01, value_coef=0.5):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.net = A2CNet(obs_dim, n_actions).to(DEVICE)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def act(self, obs, greedy=False):
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE) / 255.0
        logits, value = self.net(obs_t)
        dist = torch.distributions.Categorical(logits=logits)

        if greedy:
            action = torch.argmax(logits, dim=1)
        else:
            action = dist.sample()

        logp = dist.log_prob(action)
        entropy = dist.entropy()
        return int(action.item()), logp.squeeze(0), value.squeeze(0), entropy.squeeze(0)

    def train_step(self, logp, value, entropy, reward, next_value, done):
        # reward, done are python floats (or tensors). Make tensors on DEVICE:
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=DEVICE)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=DEVICE)

        target = reward_t + self.gamma * next_value * (1.0 - done_t)
        advantage = target - value

        policy_loss = -(logp * advantage.detach())
        value_loss = advantage.pow(2)
        entropy_loss = -entropy

        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()

    def save_checkpoint(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(
            {
                "net_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filepath,
        )
        print(f"A2C checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        if not os.path.exists(filepath):
            print(f"Checkpoint {filepath} not found")
            return False
        ckpt = torch.load(filepath, map_location=DEVICE)
        self.net.load_state_dict(ckpt["net_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"A2C checkpoint loaded from {filepath}")
        return True
