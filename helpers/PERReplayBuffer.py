import numpy as np
class PERReplayBuffer:
    def __init__(self, capacity, alpha=0.6, eps=1e-3):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, transition, priority=None):
        if priority is None:
            max_prio = self.priorities.max() if self.buffer else 1.0
            priority = max_prio
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        return samples, indices

    def update_priorities(self, indices, td_errors):
        td_errors = np.abs(td_errors) + self.eps
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = float(err)