from collections import deque

import numpy as np
import random
import torch


class DQN:
    """A deep Q-learning network using PyTorch."""

    def __init__(self, s_dim: int, a_dim: int, lr: float):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(s_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, a_dim),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.s_dim = s_dim
        self.a_dim = a_dim

    def _to_one_hot(self, vals, dim) -> torch.Tensor:
        """Convert integer values to one-hot vectors of size dim.

        Args:
            vals: List of integer values.
            dim: Dimension of one-hot vectors.

        Returns:
            One-hot encoding of vals list.
        """
        return torch.nn.functional.one_hot(vals, num_classes=dim).float()

    def compute_q(self, states, actions) -> torch.Tensor:
        """Compute Q-values of given states and actions.

        Args:
            states: Tensor of states.
            actions: List of actions.

        Returns:
            Predicted Q-values for given states and actions.
        """
        states = torch.FloatTensor(states)
        q_preds: torch.Tensor = self.model(states)
        actions_oneshot = self._to_one_hot(actions, self.a_dim)
        q_preds_selected = torch.sum(q_preds * actions_oneshot, dim=-1)
        return q_preds_selected

    def compute_max_q(self, states) -> np.ndarray:
        """Compute max_a Q(s, a) for each input state s in states.

        Args:
            states: Tensor of states.

        Returns:
            Maximum Q-values for each state.
        """
        states = torch.FloatTensor(states)
        q_vals = self.model(states).cpu().data.numpy()
        max_q = np.max(q_vals, axis=1)
        return max_q

    def compute_argmax_q(self, state) -> np.intp:
        """Compute argmax_a Q(s, a) for input state s.

        Args:
            state: Single state tensor.

        Returns:
            Best (greedy) action given current state.
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        q_val = self.model(state).cpu().data.numpy()
        greedy_action = np.argmax(q_val.flatten())
        return greedy_action

    def train(self, states, actions, targets) -> np.ndarray:
        """Train DQN on provided states, actions, and targets (rewards).

        Args:
            states: Tensor of states.
            actions: Tensor of actions.
            targets: Target reward values for given actions and states.

        Returns:
            List of training losses.
        """
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        targets = torch.FloatTensor(targets)

        q_preds_selected = self.compute_q(states, actions)

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(q_preds_selected, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().data.numpy()


class ReplayBuffer:
    """Simple replay buffer."""

    def __init__(self, max_len: int):
        self.buffer = deque()
        self.number = 0
        self.max_len = max_len

    def append(self, exp):
        """Append experience to replay buffer.

        Args:
            exp: The experience to remove from the buffer.
        """
        self.buffer.append(exp)
        self.number += 1

    def pop(self):
        """Remove oldest experience from replay buffer.

        Raises:
            IndexError: Error raised when buffer is already empty.
        """
        while self.number > self.max_len:
            self.buffer.popleft()
            self.number -= 1

    def sample(self, batch_size: int) -> list:
        """Sample a minibatch of experiences.

        Args:
            batch_size: Size of minibatch to sample.

        Returns:
            Minimatch of size min(batch_size, self.number).
        """
        return random.sample(self.buffer, min(batch_size, self.number))
