import gym
import numpy as np


def is_space_valid(space):
    if space['type'] not in ('int', 'float'):
        return False
    if not isinstance(space['shape'], tuple):
        return False
    if not all(isinstance(x, int) and x > 0 for x in space['shape']):
        return False
    if 'num_values' in space:
        if space['type'] != 'int':
            return False
        if not isinstance(space['num_values'], int):
            return False
        if space['num_values'] <= 1:
            return False
    if ('min_value' in space) is not ('max_value' in space):
        return False
    if 'min_value' in space:
        if space['type'] != 'float':
            return False
        if not isinstance(space['min_value'], np.ndarray):
            return False
        if space['min_value'].dtype not in (np.float32, np.float64):
            return False
        if not isinstance(space['max_value'], np.ndarray):
            return False
        if space['max_value'].dtype != space['min_value'].dtype:
            return False
        if (space['min_value'] >= space['max_value']).any():
            return False
    return True


def is_value_valid(value, space):
    assert is_space_valid(space)
    if not isinstance(value, np.ndarray):
        return False
    if np.isnan(value).any() or np.isinf(value).any():
        return False
    if space['type'] == 'int' and value.dtype not in (np.int32, np.int64):
        return False
    if space['type'] == 'float' and value.dtype not in (np.float32, np.float64):
        return False
    if value.shape != space['shape']:
        return False
    if 'num_values' in space:
        if (value < 0).any() or (value >= space['num_values']).any():
            return False
    if 'min_value' in space:
        if (value < space['min_value']).any() or (value > space['max_value']).any():
            return False
    return True


def convert_gym_to_dict_space(space):
    """Convert gym space to dict format."""
    if isinstance(space, gym.spaces.Discrete):
        space = dict(type='int', shape=(), num_values=int(space.n))

    elif isinstance(space, gym.spaces.MultiBinary):
        raise NotImplementedError

    elif isinstance(space, gym.spaces.MultiDiscrete):
        if (space.nvec == space.nvec.item(0)).all():
            space = dict(
                type='int', shape=tuple(map(int, space.nvec.shape)),
                num_values=int(space.nvec.item(0))
            )
        else:
            raise NotImplementedError

    elif isinstance(space, gym.spaces.Box):
        if (space.low < -10e6).all() and (space.high > 10e6).all():
            space = dict(type='float', shape=tuple(map(int, space.shape)))
        else:
            low = np.where(space.low < -10e6, -np.inf, space.low)
            high = np.where(space.high > 10e6, np.inf, space.high)
            space = dict(
                type='float', shape=tuple(map(int, space.shape)), min_value=low,
                max_value=high
            )

    assert is_space_valid(space)
    return space
