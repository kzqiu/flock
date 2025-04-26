import collections
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class ReplayBuffer:
    """ Simple replay buffer. """

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, exp):
        """Append experience to replay buffer.

        Args:
            exp: The experience to remove from the buffer.
        """
        self.buffer.append(exp)

    def sample(self, batch_size: int) -> tuple:
        """Sample a minibatch of experiences.

        Args:
            batch_size: Size of minibatch to sample.

        Returns:
            Minimatch of size min(batch_size, len(self.buffer)).
        """
        raw_batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, act, r, next_obs, done = map(np.stack, zip(*raw_batch))

        return (
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(act, dtype=torch.float32),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1), # Ensure reward is [batch_size, 1]
            torch.tensor(next_obs, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32).unsqueeze(1) # Ensure done is [batch_size, 1]
        )


class SharedActor(nn.Module):
    """ Shared actor in collaborative setting with homogeneous agents. """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128, lr: float = 1e-4):
        """
        Initialize shared actor.

        Args:
            obs_dim (int): Local observation size.
            act_dim (int): Local actor action size.
            hidden_dim (int): Hidden layer size.
            lr (float): Learning rate for optimizer.
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # network to map states to actions
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh(),
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, state: Tensor) -> Tensor:
        """
        Forward pass for shared actor.

        Args:
            state (torch.Tensor): Batch of local observations.

        Returns:
            torch.Tensor: 
        """
        # TODO: validate input shapes
        return self.model(state)


class CentralCritic(nn.Module):
    """ Centralized critic for collaborative setting. """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128, lr: float = 1e-4):
        """
        Initialize central critic.

        Args:
            obs_dim (int): Global observation size.
            act_dim (int): Global action size.
            hidden_dim (int): Hidden layer size.
            lr (float): Learning rate for optimizer.
        """
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def forward(self, state: Tensor, action: Tensor) -> torch.Tensor:
        """
        Forward pass for central critic.

        Args:
            state (torch.Tensor): Batch of global observations.
            action (torch.Tensor): Batch of global actions.

        Returns:
            torch.Tensor: 
        """
        # TODO: validate input shapes
        combined_input = torch.cat([state, action], dim=1)
        q_val = self.model(combined_input)
        return q_val


class SHMADDPG:
    """ Shared-weight Homogeneous Multi-Agent DDPG. """

    def __init__(
        self,
        n_agents: int,
        local_obs_dim: int,
        global_obs_dim: int,
        agent_act_dim: int,
        gamma: float,
        tau: float,
        buffer_size: int,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize algorithm framework.

        Args:
            n_agents (int): 
            local_obs_dim (int):
            global_obs_dim (int):
            agent_act_dim (int):
            gamma (float):
            tau (float):
            buffer_size (int):
            device (torch.device): 
        """
        self.n_agents = n_agents
        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.agent_obs_dim = local_obs_dim + global_obs_dim
        self.total_obs_dim = local_obs_dim * n_agents + global_obs_dim
        self.agent_act_dim = agent_act_dim
        self.total_act_dim = agent_act_dim * n_agents
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        self.actor = SharedActor(self.agent_obs_dim, self.agent_act_dim).to(device)
        self.critic = CentralCritic(self.total_obs_dim, self.total_act_dim).to(device)

        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)

        for p in self.actor_target.parameters():
            p.requires_grad = False

        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_actions(self, local_state: np.ndarray, global_state: np.ndarray, noise_stddev: float = 0.1, explore: bool = False) -> np.ndarray:
        """
        Select actions given agent states.
            
        Args:
            local_state (np.ndarray): Local states of agents, (n_agents, self.local_obs_dim).
            global_state (np.ndarray): Global states, (self.global_obs_dim)

        Returns:
            np.ndarray: Chosen actions, (n_agents, self.agent_act_dim)
        """
        actions = []
        self.actor.eval()

        with torch.no_grad():
            for i in range(self.n_agents):
                agent_obs_tensor = torch.tensor(np.concat((local_state[i], global_state)), dtype=torch.float32).unsqueeze(0).to(self.device)
                action = self.actor(agent_obs_tensor).squeeze(0).cpu().numpy()

                if explore:
                    noise = np.random.normal(0, noise_stddev, size=self.agent_act_dim)
                    action += noise
                    action = np.clip(action, -1.0, 1.0)

                actions.append(action)

        self.actor.train()

        return np.array(actions)

    def update(self, batch_size: int = 1):
        if len(self.replay_buffer) < batch_size:
            return

        obs, act, r, next_obs, done = self.replay_buffer.sample(batch_size)

        obs = obs.to(self.device)
        act = act.to(self.device)
        r = r.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)

        # critic update
        with torch.no_grad():
            next_actions = []
            global_next_obs = next_obs[:, -self.global_obs_dim:]

            for i in range(self.n_agents):
                start_idx = i * self.local_obs_dim
                end_idx = (i + 1) * self.local_obs_dim
                local_next_obs = next_obs[:, start_idx:end_idx]
                total_next_obs = torch.cat((local_next_obs, global_next_obs), dim=1)
                next_action_i = self.actor_target(total_next_obs)
                next_actions.append(next_action_i)

            next_action_tensor = torch.cat(next_actions, dim=1)
            target_q_vals = self.critic_target(next_obs, next_action_tensor)
            target_q = r + self.gamma * target_q_vals * (1.0 - done)

        curr_q = self.critic(obs, act)
        critic_loss = torch.nn.functional.mse_loss(curr_q, target_q)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0) # optional
        self.critic.optimizer.step()

        # actor update
        curr_actions = []
        global_obs = obs[:, -self.global_obs_dim:]
        
        for i in range(self.n_agents):
            start_idx = i * self.local_obs_dim
            end_idx = (i + 1) * self.local_obs_dim
            local_obs = obs[:, start_idx:end_idx]
            total_obs = torch.cat((local_obs, global_obs), dim=1)
            curr_action_i = self.actor(total_obs)
            curr_actions.append(curr_action_i)

        curr_action_tensor = torch.cat(curr_actions, dim=1)
        actor_loss = -self.critic(obs, curr_action_tensor).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0) # optional
        self.actor.optimizer.step()

        self.soft_update(self.critic_target, self.critic, self.tau)
        self.soft_update(self.actor_target, self.actor, self.tau)


    def soft_update(self, target_net, source_net, tau):
        """ Helper function for Polyak averaging. """
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

def train():
    pass


if __name__ == "__main__":
    pass
