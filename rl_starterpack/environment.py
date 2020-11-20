import logging

import numpy as np

from . import utils


class Environment(object):
    """Environment base class."""

    def __init__(self, state_space, action_space, max_timesteps):
        """Environment constructor.

        Args:
            state_space (dict): State space.
            action_space (dict): Action space.
            max_timesteps (int): Abort episode (terminal=2) after given number
                of timesteps.
        """
        assert utils.is_space_valid(space=state_space)
        assert utils.is_space_valid(space=action_space)

        self.state_space = state_space
        self.action_space = action_space
        self.max_timesteps = int(max_timesteps)

        self.is_terminated = True

    def close(self):
        """Close the environment."""
        pass

    def _reset(self, **kwargs):
        """Reset the environment and return the initial state.

        Returns:
            np.ndarray: Initial state.
        """
        raise NotImplementedError

    def _step(self, action, **kwargs):
        """Execute action to advance environment by one timestep and return
        next state plus reward and terminal information.

        Args:
            action (np.ndarray): Action.

        Returns:
            np.ndarray, float, bool: Next state, reward, terminal.
        """
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def reset(self, **kwargs):
        """Reset the environment and return the initial state.

        Returns:
            np.ndarray: Initial state.
        """
        if not self.is_terminated:
            logging.warning("Reset environment which has not yet terminated.")

        self.is_terminated = False
        self.timestep = 0

        state = self._reset(**kwargs)

        state = np.asarray(state)
        assert utils.is_value_valid(value=state, space=self.state_space)

        return state

    def step(self, action, **kwargs):
        """Execute action to advance environment by one timestep and return
        next state plus reward and terminal information.

        Args:
            action (np.ndarray): Action.

        Returns:
            np.ndarray, float, int: Next state, reward, terminal.
        """
        assert not self.is_terminated

        action = np.asarray(action)
        assert utils.is_value_valid(value=action, space=self.action_space)

        state, reward, terminal = self._step(action=action, **kwargs)

        state = np.asarray(state)
        assert utils.is_value_valid(value=state, space=self.state_space)

        reward = float(reward)
        terminal = bool(terminal)

        self.timestep += 1
        self.is_terminated = terminal

        if not terminal and self.timestep >= self.max_timesteps:
            self.is_terminated = True
            terminal = 2
        else:
            terminal = int(terminal)

        return state, reward, terminal
