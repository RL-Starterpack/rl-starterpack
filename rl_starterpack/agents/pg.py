import numpy as np
import torch.nn
from torch.distributions import Categorical, Normal

from .. import Agent


class PG(Agent):
    """Policy gradient."""

    def __init__(
            self, state_space, action_space,
            network_fn, learning_rate,
            discount=1.0, normalize_returns=False
    ):
        """PG constructor.

        Args:
            state_space (dict): State space.
            action_space (dict): Action space.
            network_fn (callable[-> torch.nn.Module]): Policy network
                constructor, the policy network is required to yield a rank-1
                output tensor with dimension corresponding to the number of
                actions for the application environment, usually accomplished
                with a linear layer (categorical / continuous action space):
                    torch.nn.Linear(in_features=hidden_size, out_features=env.action_space['num_values'])
                    torch.nn.Linear(in_features=hidden_size, out_features=env.action_space['shape'][0])
            learning_rate (float > 0.0): Update learning rate.
            discount (0.0 <= float <= 1.0): Return discount factor.
            normalize_returns (bool): Whether to normalize returns across the
                batch.
        """
        super().__init__(state_space=state_space, action_space=action_space)

        # Static parameters
        if self.action_space['type'] == 'int':
            assert action_space['shape'] == ()
            self.is_continuous = False
            self.num_actions = self.action_space['num_values']
        else:
            assert len(action_space['shape']) == 1
            self.is_continuous = True
            self.num_actions = self.action_space['shape'][0]
            self.min_value = self.action_space.get('min_value')
            self.max_value = self.action_space.get('max_value')

        self.discount = discount
        assert 0.0 <= self.discount <= 1.0

        self.normalize_returns = normalize_returns

        # Network
        self.network = network_fn()
        assert isinstance(self.network, torch.nn.Module)
        params = list(self.network.parameters())
        if self.is_continuous:
            self.log_stddev = torch.zeros(self.num_actions)
            params.append(self.log_stddev)
        assert learning_rate > 0.0
        self.optimizer = torch.optim.Adam(params=params, lr=learning_rate)

        # Episode buffer
        self.episode_buffer = list()

    def _act(self, state, evaluation):
        # Evaluation mode
        self.network.eval()
        with torch.no_grad():
            state = np.expand_dims(state, axis=0)
            dtype = (torch.float32 if self.state_space['type'] == 'float' else torch.int64)
            state = torch.tensor(state, dtype=dtype)
            logits_mean = self.network(state)
            assert logits_mean.shape[1] == self.num_actions

            if evaluation:
                # Maximum likelihood action
                if self.is_continuous:
                    # Gaussian: mean
                    action = logits_mean.detach().numpy()
                    if self.min_value is not None:
                        action = np.maximum(action, self.min_value)
                    if self.max_value is not None:
                        action = np.minimum(action, self.max_value)
                else:
                    # Categorical: highest-probability action
                    logits_mean = np.squeeze(logits_mean.detach().numpy(), axis=0)
                    (max_actions,) = (logits_mean == logits_mean.max(axis=-1)).nonzero()
                    action = np.random.choice(max_actions)

            else:
                # Sample action
                if self.is_continuous:
                    distribution = Normal(loc=logits_mean, scale=torch.exp(self.log_stddev))
                    action = distribution.sample().detach().numpy()
                    if self.min_value is not None:
                        action = np.maximum(action, self.min_value)
                    if self.max_value is not None:
                        action = np.minimum(action, self.max_value)
                else:
                    distribution = Categorical(logits=logits_mean)
                    action = distribution.sample().detach().numpy()

        return np.squeeze(action, axis=0)

    def _observe(self, state, action, reward, terminal, next_state):
        # Store timestep in episode buffer
        self.episode_buffer.append((state, action, reward, terminal))

        # Check whether episode terminated
        if terminal == 0:
            return False

        # Retrieve and clear episode buffer
        state, action, reward, terminal = map(np.asarray, zip(*self.episode_buffer))
        self.episode_buffer.clear()

        # Reward computation: reward to go
        rollout_value = self.reward_to_go(reward, terminal, self.discount)
        assert rollout_value.shape == reward.shape

        # Normalize returns
        if self.normalize_returns:
            rollout_value -= rollout_value.mean()
            rollout_value /= max(rollout_value.std(), 1e-3)

        # Make tensors
        dtype = (torch.float32 if self.state_space['type'] == 'float' else torch.int64)
        state = torch.tensor(state, dtype=dtype)
        dtype = (torch.float32 if self.action_space['type'] == 'float' else torch.int64)
        action = torch.tensor(action, dtype=dtype)
        rollout_value = torch.tensor(rollout_value, dtype=torch.float32)

        # Action log probabilities (training mode)
        self.network.train()
        logits_mean = self.network(state)
        assert logits_mean.shape[1] == self.num_actions
        if self.is_continuous:
            distribution = Normal(loc=logits_mean, scale=torch.exp(self.log_stddev))
        else:
            distribution = Categorical(logits=logits_mean)
        log_prob = distribution.log_prob(action)

        surrogate_loss = self.surrogate_loss(rollout_value, log_prob)
        assert surrogate_loss.shape == torch.Size([])  # Assert that the implementation returned a scalar
        self.optimizer.zero_grad()
        surrogate_loss.backward()
        self.optimizer.step()

        return True

    @staticmethod
    def reward_to_go(reward, terminal, discount):
        num_timesteps = reward.shape[0]
        rollout_value = reward
        for n in range(num_timesteps - 2, -1, -1):
            if terminal[n] == 0:
                rollout_value[n] += discount * rollout_value[n + 1]
        return rollout_value

    @staticmethod
    def surrogate_loss(rollout_value, log_prob):
        # Surrogate loss:  -(R[t:] * log pi(s_t, a_t))
        return -(rollout_value * log_prob).mean()
