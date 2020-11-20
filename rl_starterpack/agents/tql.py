import numpy as np
from random import random
from collections import defaultdict

from .. import Agent


class TQLBase(Agent):
    """Base class for tabular Q-learning.

    Relies on sub-classes to implement an exploration policy and a deterministic Q-learning policy."""

    def __init__(self, state_space, action_space, learning_rate, discount=1.0):
        """TQL constructor.

        Args:
            state_space (dict): State space.
            action_space (dict): Action space.
            learning_rate (float > 0.0): Update learning rate.
            discount (0.0 <= float <= 1.0): Return discount factor.
        """
        assert action_space['type'] == 'int'
        assert action_space['shape'] == ()
        assert 'num_values' in action_space

        super().__init__(state_space=state_space, action_space=action_space)

        self.discount = float(discount)
        assert 0.0 <= self.discount <= 1.0

        self.learning_rate = float(learning_rate)
        assert self.learning_rate > 0.0

        # Map from states to values for each action
        # Initialise each Q-value to -np.inf to represent an unexplored state/action pair
        self.q_table = defaultdict(self._initialise_action_values)

    def _act(self, state, evaluation):
        # Get Q-values associated with state
        q_values = self.q_table[self._hashable_state(state)]

        # Choose whether to explore or not
        action = self.exploration_policy(evaluation, q_values)

        # If not exploring, use a q-learning policy
        if action is None:
            action = self.q_learning_policy(q_values)

        return action

    def _observe(self, state, action, reward, terminal, next_state):
        # Make states hashable
        state = self._hashable_state(state)
        next_state = self._hashable_state(next_state)

        # Temporal difference update
        self.td_update(state, action, self.td_target(reward, self.next_state_value(next_state, terminal)))

        return True

    def next_state_value(self, next_state, terminal):
        """The value of the next state.

        Args:
            next_state: The next state
            terminal (bool): Is the current state terminal?

        Returns:
            The Q-value of the next state (max over Q-values) or 0 if the current state is terminal
            or the next state has no explored actions.
        """
        # Value of next state as max over Q-values, unless terminal
        if terminal == 1:
            # Nowhere to go from terminal state, so the value of the next state is 0
            value_next_state = 0.0
        else:
            value_next_state = self.q_table[next_state].max()
            if np.isinf(value_next_state):  # State has never been visited
                # Estimate its value as 0 (this is effectively a parameter of the algorithm)
                value_next_state = 0.0
        return value_next_state

    def _initialise_action_values(self):
        """Action values for a state (initialised to negative infinity to represent unexplored actions)."""
        return np.full((self.action_space['num_values'],), fill_value=-np.inf, dtype=float)

    def _hashable_state(self, state):
        """Hashable version of the state."""
        return tuple(state.flatten().tolist())

    def exploration_policy(self, evaluation, q_values):
        """Policy that decides whether to explore or not.

        Args:
            evaluation (bool): Evaluation mode, i.e. whether to act deterministically.
            q_values (nd.array): Q-value for each possible action (-np.inf for unexplored)

        Returns:
            The action to explore when exploring or None if policy chooses not to explore.
            The action is represented as an index into the q_values array.

        """
        raise NotImplementedError()

    def q_learning_policy(self, q_values):
        """Q-learning policy that takes best action (breaking ties randomly).

        Args:
            q_values (nd.array): Q-value for each possible action (-np.inf for unexplored)

        Returns:
            chosen action (index into q_values)
        """
        raise NotImplementedError()

    def td_target(self, reward, value_next_state):
        """Calculate the temporal difference target."""
        raise NotImplementedError()

    def td_update(self, state, action, td_target):
        """Update the Q-value table using temporal difference."""
        raise NotImplementedError()

    @staticmethod
    def unexplored_actions(q_values):
        return np.isinf(q_values).nonzero()[0]

    @staticmethod
    def explored_actions(q_values):
        return (~ np.isinf(q_values)).nonzero()[0]


class TQL(TQLBase):
    """Tabular Q-learning."""

    def __init__(self, state_space, action_space, learning_rate, discount=1.0, exploration=0.0):
        """TQL constructor.

        Args:
            state_space (dict): State space.
            action_space (dict): Action space.
            learning_rate (float > 0.0): Update learning rate.
            discount (0.0 <= float <= 1.0): Return discount factor.
            exploration (0.0 <= float <= 1.0): Random exploration rate.
        """
        super().__init__(state_space=state_space,
                         action_space=action_space,
                         learning_rate=learning_rate,
                         discount=discount)

        self.exploration = float(exploration)
        assert 0.0 <= self.exploration <= 1.0

    def exploration_policy(self, evaluation, q_values):
        """Policy that decides whether to explore or not.

        Args:
            evaluation (bool): Evaluation mode, i.e. whether to act deterministically.
            q_values (nd.array): Q-value for each possible action (-np.inf for unexplored)

        Returns:
            The action to explore when exploring or None if policy chooses not to explore.
            The action is represented as an index into the q_values array.

        """
        explored = TQLBase.explored_actions(q_values)
        unexplored = TQLBase.unexplored_actions(q_values)

        if not evaluation and len(unexplored):
            # Force exploration of unexplored actions when not evaluating
            return np.random.choice(unexplored)

        elif len(explored) and (evaluation or random() >= self.exploration):
            # Don't explore (and thus instead exploit) when:
            #     * We are in evaluation mode and there are explored actions.
            #     * We didn't hit the exploration probability threshold and there are explored actions.
            return None

        else:
            # Totally random exploration
            return np.random.choice(len(q_values))

    def q_learning_policy(self, q_values):
        """Q-learning policy that takes best action (breaking ties randomly).

        Args:
            q_values (nd.array): Q-value for each possible action (-np.inf for unexplored)

        Returns:
            chosen action (index into q_values)
        """
        best_actions = (q_values == q_values.max()).nonzero()[0]
        return np.random.choice(best_actions)

    def td_target(self, reward, value_next_state):
        """Calculate the temporal difference target."""
        return reward + self.discount * value_next_state

    def td_update(self, state, action, td_target):
        """Update the Q-value table using temporal difference."""
        if np.isinf(self.q_table[state][action]):  # First update
            self.q_table[state][action] = td_target
        else:
            td = td_target - self.q_table[state][action]
            self.q_table[state][action] += self.learning_rate * td
