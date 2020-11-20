import numpy as np

from .. import Agent


class RandomAgent(Agent):
    """Random agent."""

    def _act(self, state, evaluation):
        if self.action_space['type'] == 'int':
            assert 'num_values' in self.action_space
            return np.random.randint(
                self.action_space['num_values'], size=self.action_space['shape']
            )

        elif self.action_space['type'] == 'float':
            if 'min_value' in self.action_space:
                return np.random.uniform(
                    self.action_space['min_value'], self.action_space['max_value'],
                    size=self.action_space['shape']
                )

            else:
                return np.random.normal(size=self.action_space['shape'])

    def _observe(self, state, action, reward, terminal, next_state):
        return False
