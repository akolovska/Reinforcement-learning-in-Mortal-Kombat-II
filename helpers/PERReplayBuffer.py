import numpy as np


class PERReplayBuffer:
    """
    Prioritized Experience Replay Buffer with Importance Sampling

    Features:
    - Prioritizes transitions based on TD error magnitude
    - Alpha: Controls how much prioritization is used (0 = uniform, 1 = full prioritization)
    - Beta: Controls importance sampling weight (anneals from beta_start to 1.0)
    - Epsilon: Small constant to ensure non-zero priorities
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000, eps=1e-3):
        """
        Args:
            capacity: Maximum size of the buffer
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight (typically 0.4)
            beta_frames: Number of frames over which to anneal beta to 1.0
            eps: Small constant added to priorities to avoid zero probability
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.frame = 0  # Track frames for beta annealing

    def __len__(self):
        return len(self.buffer)

    def add(self, transition, priority=None):
        """
        Add a transition to the buffer

        Args:
            transition: The experience tuple (state, action, reward, next_state, done)
            priority: Optional priority value (uses max priority if None)
        """
        if priority is None:
            # New transitions get max priority to ensure they're sampled at least once
            max_prio = self.priorities.max() if self.buffer else 1.0
            priority = max_prio

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of transitions with prioritization

        Args:
            batch_size: Number of transitions to sample

        Returns:
            samples: List of sampled transitions
            indices: Indices of sampled transitions (for priority updates)
            weights: Importance sampling weights for each sample
        """
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]

        # Calculate sampling probabilities based on priorities
        probs = prios ** self.alpha
        probs /= probs.sum()

        # Sample indices according to priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Calculate importance sampling weights with beta annealing
        # Beta anneals from beta_start to 1.0 over beta_frames
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        # Importance sampling weight: (N * P(i))^(-beta)
        # Normalized by dividing by max weight for stability
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = weights.astype(np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD errors

        Args:
            indices: Indices of transitions to update
            td_errors: TD errors for each transition (priority = |TD error| + eps)
        """
        td_errors = np.abs(td_errors) + self.eps
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = float(err)