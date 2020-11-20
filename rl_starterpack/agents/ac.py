import numpy as np
import torch.nn
from torch.distributions import Categorical, Normal

from .. import Agent


class AC(Agent):
    """Q/TD actor-critic."""

    def __init__(
        self, state_space, action_space,
        actor_fn, actor_learning_rate,
        critic_fn, critic_learning_rate,
        discount=1.0, compute_advantage=False, normalize_returns=False
    ):
        """AC constructor, Q actor-critic by default, TD actor-critic if
        compute_advantage is false.

        Args:
            state_space (dict): State space.
            action_space (dict): Action space.
            actor_fn (callable[-> torch.nn.Module]): Actor policy network
                constructor, the policy network is required to yield a rank-1
                output tensor with dimension corresponding to the number of
                actions for the application environment, usually accomplished
                with a linear layer (categorical / continuous action space):
                    torch.nn.Linear(in_features=hidden_size, out_features=env.action_space['num_values'])
                    torch.nn.Linear(in_features=hidden_size, out_features=env.action_space['shape'][0])
            critic_fn (callable[-> torch.nn.Module]): Critic value network
                constructor, the value network is required to yield a rank-1
                output tensor of dimension 1, usually accomplished with a
                linear layer:
                    torch.nn.Linear(in_features=hidden_size, out_features=1)
            actor_learning_rate (float > 0.0): Actor update learning rate.
            critic_learning_rate (float > 0.0): Critic update learning rate.
            discount (0.0 <= float <= 1.0): Return discount factor.
            compute_advantage (bool): Whether to compute advantage by
                subtracting critic value estimate (TD actor-critic).
            normalize_returns (bool): Whether to normalize returns (or
                advantages) across the batch.
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
            assert (self.min_value is None) is (self.max_value is None)

        self.discount = discount
        assert 0.0 <= self.discount <= 1.0

        self.compute_advantage = compute_advantage
        self.normalize_returns = normalize_returns

        # Actor network (equivalent to PG agent)
        self.network = actor_fn()
        assert isinstance(self.network, torch.nn.Module)
        params = list(self.network.parameters())
        if self.is_continuous:
            self.log_stddev = torch.zeros(self.num_actions)
            params.append(self.log_stddev)
        assert actor_learning_rate > 0.0
        self.optimizer = torch.optim.Adam(params=params, lr=actor_learning_rate)

        # Critic network (similar to DQN agent)
        self.critic_network = critic_fn()
        params = list(self.critic_network.parameters())
        assert critic_learning_rate > 0.0
        self.critic_optimizer = torch.optim.Adam(params=params, lr=critic_learning_rate)
        self.critic_loss = torch.nn.SmoothL1Loss()

        # Normalize returns: moving mean and stddev
        if self.normalize_returns:
            self.mean = None
            self.stddev = None

    def _act(self, state, evaluation):
        # (Equivalent to PG agent)

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
        # (Similar to PG agent, but without episode buffering)

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
        terminal = np.expand_dims(terminal, axis=0)
        terminal = torch.tensor(terminal == 1)

        # Critic observe
        self._critic_observe(state, action, reward, terminal, next_state)

        # Reward computation: state value / advantage predicted by critic (evaluation mode)
        self.critic_network.eval()
        with torch.no_grad():
            v_next = self.critic_network(next_state)
            assert v_next.shape[1] == 1
            torch.where(terminal, torch.zeros_like(v_next), v_next)
            target = reward + self.discount * v_next.squeeze(dim=1)
            if self.compute_advantage:
                # Compute advantage
                state_value = self.critic_network(state)
                assert state_value.shape[1] == 1
                target -= state_value.squeeze(dim=1)
            target = target.detach()

        # Normalize returns
        if self.normalize_returns:
            if self.mean is None:
                self.mean = target.numpy().item()
            elif self.stddev is None:
                x = target.numpy().item()
                self.mean = 0.99 * self.mean + 0.01 * x
                self.stddev = np.abs(x - self.mean)
            else:
                x = target.numpy().item()
                self.mean = 0.99 * self.mean + 0.01 * x
                self.stddev = 0.99 * self.stddev + 0.01 * np.abs(x - self.mean)
                target = (target - self.mean) / max(self.stddev, 1e-3)

        # Action log probabilities (training mode)
        self.network.train()
        logits_mean = self.network(state)
        assert logits_mean.shape[1] == self.num_actions
        if self.is_continuous:
            distribution = Normal(loc=logits_mean, scale=torch.exp(self.log_stddev))
        else:
            distribution = Categorical(logits=logits_mean)
        log_prob = distribution.log_prob(action)

        # Surrogate loss:  -( {V(s_t), A(s_t,a_t)} * log pi(s_t, a_t))
        loss = -(target * log_prob).mean()
        self.last_actor_loss_value = loss.detach().numpy().mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return True

    def _critic_observe(self, state, action, reward, terminal, next_state):
        # (Similar to DQN agent, but state instead of Q-value)

        # Temporal difference target (evaluation mode)
        self.critic_network.eval()
        with torch.no_grad():
            v_next = self.critic_network(next_state)
            assert v_next.shape[1] == 1
            v_next = torch.where(terminal.unsqueeze(dim=1), torch.zeros_like(v_next), v_next)
            td_target = reward.unsqueeze(dim=1) + self.discount * v_next
            td_target = td_target.detach()

        # Temporal difference update (training mode)
        self.critic_network.train()
        state_value = self.critic_network(state)
        assert state_value.shape[1] == 1
        loss = self.critic_loss(state_value, td_target)
        self.last_critic_loss_value = loss.detach().numpy().mean()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
