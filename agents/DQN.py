import random
from collections import deque, namedtuple
import numpy as np
import torch
from torch import optim
import torch.nn as nn
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
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
        self.replay_buffer = deque(maxlen=100_000)
        self.batch_size = 64
        self.update_target_every = 1000

    def act(self, obs, greedy=False):
        """
        If greedy=True: always pick argmax (no exploration) – used for evaluation.
        Otherwise: epsilon-greedy – used for training.
        """
        self.step_count += 1
        if not greedy:
            eps = self._get_epsilon()
            if random.random() < eps:
                return random.randrange(self.n_actions)

        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_vals = self.q_net(obs_t)
        return int(q_vals.argmax(dim=1).item())

    def _get_epsilon(self):
        # linear decay
        return max(self.epsilon_end,
                   self.epsilon - (1.0 - self.epsilon_end) * self.step_count / self.epsilon_decay)

    def store(self, *args):
        self.replay_buffer.append(Transition(*args))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        batch = Transition(*zip(*batch))

        state = torch.from_numpy(np.stack(batch.state)).float().to(DEVICE)
        action = torch.tensor(batch.action).long().to(DEVICE)
        reward = torch.tensor(batch.reward).float().to(DEVICE)
        next_state = torch.from_numpy(np.stack(batch.next_state)).float().to(DEVICE)
        done = torch.tensor(batch.done).float().to(DEVICE)

        q_values = self.q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_state).max(1)[0]
            target = reward + self.gamma * next_q * (1 - done)

        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())