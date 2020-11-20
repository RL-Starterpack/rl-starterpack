import numpy as np

from . import utils


class Agent(object):
    """Agent base class."""

    def __init__(self, state_space, action_space):
        """Agent constructor.

        Args:
            state_space (dict): State space.
            action_space (dict): Action space.
        """
        assert utils.is_space_valid(space=state_space)
        assert utils.is_space_valid(space=action_space)

        self.state_space = state_space
        self.action_space = action_space

    @classmethod
    def load(self, path):
        """Load an agent model.

        Args:
            path (str): Path to load from.
        """
        raise NotImplementedError

    def save(self, path):
        """Save the agent model.

        Args:
            path (str): Path to save to.
        """
        raise NotImplementedError

    def close(self):
        """Close the agent."""
        pass

    def _act(self, state, evaluation, **kwargs):
        """Retrieve an action given a state.

        Args:
            state (np.ndarray): State.
            evaluation (bool): Evaluation mode, i.e. whether to act
                deterministically.

        Returns:
            np.ndarray: Action.
        """
        raise NotImplementedError

    def _observe(self, state, action, reward, terminal, next_state, **kwargs):
        """Observe a training timestep and may perform an update.

        Args:
            state (np.ndarray): State.
            action (np.ndarray): Action.
            reward (float): Reward.
            terminal (int): Terminal.
            next_state (np.ndarray): Next state.

        Returns:
            bool: Whether an update was performed.
        """
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def act(self, state, evaluation=False, **kwargs):
        """Retrieve an action given a state.

        Args:
            state (np.ndarray): State.
            evaluation (bool): Evaluation mode, i.e. whether to act
                deterministically.

        Returns:
            np.ndarray: Action.
        """
        state = np.asarray(state)
        assert utils.is_value_valid(value=state, space=self.state_space)

        action = self._act(state=state, evaluation=evaluation, **kwargs)

        action = np.asarray(action)
        assert utils.is_value_valid(value=action, space=self.action_space)

        return action

    def observe(self, state, action, reward, terminal, next_state, **kwargs):
        """Observe a training timestep and may perform an update.

        Args:
            state (np.ndarray): State.
            action (np.ndarray): Action.
            reward (float): Reward.
            terminal (int): Terminal.
            next_state (np.ndarray): Next state.

        Returns:
            bool: Whether an update was performed.
        """
        state = np.asarray(state)
        assert utils.is_value_valid(value=state, space=self.state_space)

        action = np.asarray(action)
        assert utils.is_value_valid(value=action, space=self.action_space)

        reward = float(reward)

        terminal = int(terminal)
        assert terminal in (0, 1, 2)

        next_state = np.asarray(next_state)
        assert utils.is_value_valid(value=next_state, space=self.state_space)

        is_update = self._observe(
            state=state, action=action, reward=reward, terminal=terminal,
            next_state=next_state, **kwargs
        )
        is_update = bool(is_update)

        return is_update
