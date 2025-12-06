import random
from collections import deque, namedtuple
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from agents.DQN import MLP
from helpers.PERReplayBuffer import PERReplayBuffer
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDQNAgent:
    def __init__(self, obs_dim, n_actions, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=500000,
                 use_per=False):

        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_count = 0
        self.use_per = use_per

        self.q_net = MLP(obs_dim, n_actions).to(DEVICE)
        self.target_net = MLP(obs_dim, n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.batch_size = 64
        self.update_target_every = 1000
        self.replay_capacity = 100_000

        if self.use_per:
            self.replay_buffer = PERReplayBuffer(self.replay_capacity)
        else:
            self.replay_buffer = deque(maxlen=self.replay_capacity)

    def act(self, obs, greedy=False):
        """
        If greedy=True: always pick argmax (no exploration) – for evaluation.
        Otherwise: epsilon-greedy – for training.
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
        return max(self.epsilon_end,
                   self.epsilon - (1.0 - self.epsilon_end) * self.step_count / self.epsilon_decay)

    def store(self, *args):
        tr = Transition(*args)
        if self.use_per:
            self.replay_buffer.add(tr)
        else:
            self.replay_buffer.append(tr)

    def train_step(self):
        if self.use_per:
            if len(self.replay_buffer) < self.batch_size:
                return
            batch, indices = self.replay_buffer.sample(self.batch_size)
            batch = Transition(*zip(*batch))
        else:
            if len(self.replay_buffer) < self.batch_size:
                return
            batch = random.sample(self.replay_buffer, self.batch_size)
            batch = Transition(*zip(*batch))
            indices = None  # not used

        state = torch.from_numpy(np.stack(batch.state)).float().to(DEVICE)
        action = torch.tensor(batch.action).long().to(DEVICE)
        reward = torch.tensor(batch.reward).float().to(DEVICE)
        next_state = torch.from_numpy(np.stack(batch.next_state)).float().to(DEVICE)
        done = torch.tensor(batch.done).float().to(DEVICE)

        # Q(s, a)
        q_values = self.q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)

        # DDQN: use online net for argmax, target net for value
        with torch.no_grad():
            next_q_online = self.q_net(next_state)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)  # (B, 1)
            next_q_target = self.target_net(next_state)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)
            target = reward + self.gamma * next_q * (1 - done)

        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # update PER priorities (if enabled)
        if self.use_per and indices is not None:
            with torch.no_grad():
                td_errors = (q_values - target).detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors)

        # update target network
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())