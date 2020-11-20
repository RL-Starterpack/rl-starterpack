import numpy as np
from random import random
import torch.nn

from .. import Agent


class DQN(Agent):
    """Deep Q-network."""

    def __init__(self, state_space, action_space, network_fn, learning_rate, discount=1.0, exploration=0.0,
                 target_network_update_frequency=1, memory=None, batch_size=None, update_frequency=1,
                 update_start=None):
        """DQN constructor.

        Args:
            state_space (dict): State space.
            action_space (dict): Action space.
            network_fn (callable[-> torch.nn.Module]): Q-network constructor,
                the Q-network is required to yield a rank-1 output tensor with
                dimension corresponding to the number of actions for the
                application environment, usually accomplished with a linear
                layer:
                    torch.nn.Linear(in_features=hidden_size, out_features=env.action_space['num_values'])
            learning_rate (float > 0.0): Update learning rate.
            discount (0.0 <= float <= 1.0): Return discount factor.
            exploration (0.0 <= float <= 1.0): Random exploration rate.
            target_network_update_frequency (int >= 1): Frequency of target
                network update with respect to main network updates, 1 implies
                no separate target network.
            memory (int >= batch_size): Replay memory capacity, None implies no
                replay memory.
            batch_size (int >= 1): Update batch size, only valid if memory is
                specified.
            update_frequency (1 <= int <= batch_size): Frequency of main
                network updates, only valid if memory is specified.
            update_start (int >= batch_size): Number of timesteps to collect
                experience to fill memory before first update, only valid if
                memory is specified.
        """
        assert action_space['type'] == 'int'
        assert action_space['shape'] == ()
        assert 'num_values' in action_space

        super().__init__(state_space=state_space, action_space=action_space)

        self.discount = float(discount)
        assert 0.0 <= self.discount <= 1.0

        self.exploration = float(exploration)
        assert 0.0 <= self.exploration <= 1.0

        self.num_actions = self.action_space['num_values']

        # Network
        self.network = network_fn()
        assert isinstance(self.network, torch.nn.Module)
        params = list(self.network.parameters())
        assert learning_rate > 0.0
        self.optimizer = torch.optim.Adam(params=params, lr=learning_rate)
        self.loss = torch.nn.SmoothL1Loss()

        # Target network
        if target_network_update_frequency == 1:
            self.target_network = self.network
            self.until_next_target_update = None
        else:
            self.target_network = network_fn()
            self.target_network.load_state_dict(self.network.state_dict())
            self.target_frequency = target_network_update_frequency
            assert self.target_frequency >= 1
            self.until_next_target_update = target_network_update_frequency

        if memory is not None:
            # Batch size, update frequency and start
            self.batch_size = batch_size
            assert self.batch_size >= 1
            self.update_frequency = update_frequency
            assert 1 <= self.update_frequency <= self.batch_size
            if update_start is None:
                self.until_next_update = self.batch_size
            else:
                assert update_start >= self.batch_size
                self.until_next_update = max(update_start, self.update_frequency)

            # Replay memory
            self.memory_capacity = memory
            assert self.memory_capacity >= self.batch_size

            self.memory_index = 0
            dtype = (np.float32 if self.state_space['type'] == 'float' else np.int64)
            self.state_memory = np.zeros(
                shape=((self.memory_capacity,) + self.state_space['shape']), dtype=dtype
            )
            dtype = (np.float32 if self.action_space['type'] == 'float' else np.int64)
            self.action_memory = np.zeros(
                shape=((self.memory_capacity,) + self.action_space['shape']), dtype=dtype
            )
            self.reward_memory = np.zeros(shape=(self.memory_capacity,), dtype=np.float32)
            self.terminal_memory = np.zeros(shape=(self.memory_capacity,), dtype=np.byte)

        elif batch_size is not None:
            raise ValueError('Argument "batch_size" only valid if "memory" specified.')
        elif update_frequency != 1:
            raise ValueError('Argument "update_frequency" only valid if "memory" specified.')
        elif update_start is not None:
            raise ValueError('Argument "update_start" only valid if "memory" specified.')
        else:
            self.update_frequency = 1
            self.until_next_update = 1
            self.memory_capacity = None

    def _act(self, state, evaluation):
        if evaluation or random() >= self.exploration:
            # Deterministic Q-learning policy using main network (evaluation mode)
            self.network.eval()
            with torch.no_grad():
                state = np.expand_dims(state, axis=0)
                dtype = (torch.float32 if self.state_space['type'] == 'float' else torch.int64)
                state = torch.tensor(state, dtype=dtype)
                q_values = self.network(state)
                assert q_values.shape[1] == self.num_actions
                q_values = np.squeeze(q_values.detach().numpy(), axis=0)
            (max_actions,) = (q_values == q_values.max(axis=-1)).nonzero()
            return np.random.choice(max_actions)

        else:
            # Random exploration
            return np.random.randint(self.num_actions)

    def _observe(self, state, action, reward, terminal, next_state):
        if self.memory_capacity is not None:
            # Add experience to replay memory
            self.state_memory[self.memory_index % self.memory_capacity] = state
            self.action_memory[self.memory_index % self.memory_capacity] = action
            self.reward_memory[self.memory_index % self.memory_capacity] = reward
            self.terminal_memory[self.memory_index % self.memory_capacity] = terminal
            self.memory_index += 1

        # Check whether next update should be performed
        self.until_next_update -= 1
        if self.until_next_update > 0:
            return False
        self.until_next_update = self.update_frequency

        # Infrequent target update
        if self.until_next_target_update is not None:
            self.until_next_target_update -= 1
            if self.until_next_target_update <= 0:
                self.until_next_target_update = self.target_frequency
                self.target_network.load_state_dict(self.network.state_dict())

        if self.memory_capacity is None:
            # Batch and make tensors
            state = np.expand_dims(state, axis=0)
            dtype = (torch.float32 if self.state_space['type'] == 'float' else torch.int64)
            state = torch.tensor(state, dtype=dtype)
            next_state = np.expand_dims(next_state, axis=0)
            next_state = torch.tensor(next_state, dtype=dtype)
            action = np.expand_dims(action, axis=0)
            dtype = (torch.float32 if self.action_space['type'] == 'float' else torch.int64)
            action = torch.tensor(action, dtype=dtype)
            reward = np.expand_dims(reward, axis=0)
            reward = torch.tensor(reward, dtype=torch.float32)
            terminal = torch.tensor(np.expand_dims(terminal == 1, axis=0))

        else:
            # Random indices to retrieve from memory (skip current timestep, drop abort-terminals)
            memory_size = min(self.memory_capacity, self.memory_index)
            indices = np.random.randint(1, memory_size, size=self.batch_size)
            indices = (self.memory_index - indices) % self.memory_capacity
            indices = indices[np.where(self.terminal_memory[indices] != 2)]

            # Retrieve batch from replay memory
            state = torch.tensor(self.state_memory[indices])
            action = torch.tensor(self.action_memory[indices])
            reward = torch.tensor(self.reward_memory[indices])
            terminal = torch.tensor(self.terminal_memory[indices] == 1)
            next_state = torch.tensor(self.state_memory[(indices + 1) % self.memory_capacity])

        # Temporal difference target using target network (evaluation mode)
        self.target_network.eval()
        with torch.no_grad():
            q_values = self.target_network(next_state)
            assert q_values.shape[1] == self.num_actions
            v_next = q_values.max(dim=-1).values
            v_next = torch.where(terminal, torch.zeros_like(v_next), v_next)
            td_target = reward + self.discount * v_next
            td_target = td_target.detach()

        # Temporal difference update of main network (training mode)
        self.network.train()
        q_values = self.network(state)
        assert q_values.shape[1] == self.num_actions
        q_value = torch.gather(q_values, dim=-1, index=action.unsqueeze(dim=-1)).squeeze(dim=-1)
        loss = self.loss(q_value, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return True
