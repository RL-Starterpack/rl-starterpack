import gym
import time

from . import Environment, utils, vis_utils


class OpenAIGym(Environment):
    """OpenAI Gym (https://gym.openai.com/).

    More information: https://github.com/openai/gym/blob/master/gym/envs/__init__.py
    """

    def __init__(self, level, max_timesteps, **kwargs):
        """Gym environment constructor.

        Args:
            level (str | gym.Env): Gym id, preferably without "-v" suffix;
                gym.Env instance or subclass also valid.
            max_timesteps (int): Abort episode (terminal=2) after given number
                of timesteps.
        """
        if isinstance(level, gym.Env):
            # Level as gym.Env instance
            assert len(kwargs) == 0
            self.level = level

        elif isinstance(level, type) and issubclass(level, gym.Env):
            # Level as gym.Env subclass
            self.level = level(**kwargs)

        else:
            # Level as gym id
            if level not in gym.envs.registry.env_specs:
                # Register version without max_episode_steps
                level = max(
                    [x for x in gym.envs.registry.env_specs if x.startswith(level + '-v')],
                    key=(lambda x: int(x[x.rindex('-v') + 2:]))
                )
                version = int(level[level.rindex('-v') + 2:])
                new_level = level[:level.rindex('-v')] + '-v' + str(version + 1)
                entry_point = gym.envs.registry.env_specs[level].entry_point
                reward_threshold = gym.envs.registry.env_specs[level].reward_threshold
                nondeterministic = gym.envs.registry.env_specs[level].nondeterministic
                _kwargs = dict(gym.envs.registry.env_specs[level]._kwargs)
                gym.register(
                    id=new_level, entry_point=entry_point, reward_threshold=reward_threshold,
                    nondeterministic=nondeterministic, max_episode_steps=None, kwargs=_kwargs
                )
                level = new_level

            # Make gym level
            self.level = gym.make(id=level, **kwargs)

        # State/action space
        state_space = utils.convert_gym_to_dict_space(space=self.level.observation_space)
        action_space = utils.convert_gym_to_dict_space(self.level.action_space)

        super().__init__(
            state_space=state_space, action_space=action_space, max_timesteps=max_timesteps
        )

    def __str__(self):
        return 'Gym({})'.format(self.level.spec.id)

    def close(self):
        """Close the environment."""
        self.level.close()

    def _reset(self):
        """Reset the environment and return the initial state.

        Returns:
            np.ndarray: Initial state.
        """
        return self.level.reset()

    def _step(self, action):
        """
        Execute action to advance environment by one timestep and return
        next state plus reward and terminal information.

        Args:
            action (np.ndarray): Action.

        Returns:
            np.ndarray, float, bool: Next state, reward, terminal.
        """
        if self.action_space['shape'] == ():
            action = action.item()
        state, reward, terminal, _ = self.level.step(action)
        return state, reward, terminal

    def render(self, ipython_display=None, sleep=0.0):
        """Render the environment.
        Args:
            ipython_display (IPython.display): IPython display.
            sleep (float): Sleep slowdown in seconds.
        """
        time.sleep(sleep)
        ipython_display.clear_output(wait=True)
        self.level.render(mode='human')
