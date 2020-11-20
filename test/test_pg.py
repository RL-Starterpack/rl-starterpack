import unittest

import torch.nn

from rl_starterpack import OpenAIGym, PG, experiment


class TestPG(unittest.TestCase):

    def test_cartpole(self):
        env = OpenAIGym(level='CartPole', max_timesteps=100)
        hidden_size = 16
        network_fn = (lambda: torch.nn.Sequential(
            torch.nn.Linear(in_features=env.state_space['shape'][0], out_features=hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=hidden_size, out_features=env.action_space['num_values'])
        ))
        agent = PG(
            state_space=env.state_space, action_space=env.action_space,
            network_fn=network_fn, learning_rate=1e-3
        )
        experiment.train(agent=agent, env=env, num_episodes=10)
        experiment.evaluate(agent=agent, env=env, num_episodes=10)
        agent.close()
        env.close()

    def test_pendulum(self):
        env = OpenAIGym(level='Pendulum', max_timesteps=100)
        hidden_size = 16
        network_fn = (lambda: torch.nn.Sequential(
            torch.nn.Linear(in_features=env.state_space['shape'][0], out_features=hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=hidden_size, out_features=env.action_space['shape'][0])
        ))
        agent = PG(
            state_space=env.state_space, action_space=env.action_space,
            network_fn=network_fn, learning_rate=1e-3,
            discount=0.95, normalize_returns=True
        )
        experiment.train(agent=agent, env=env, num_episodes=10)
        experiment.evaluate(agent=agent, env=env, num_episodes=10)
        agent.close()
        env.close()
