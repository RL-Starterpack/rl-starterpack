import unittest

import torch.nn

from rl_starterpack import AC, OpenAIGym, experiment


class TestAC(unittest.TestCase):

    def test_cartpole(self):
        env = OpenAIGym(level='CartPole', max_timesteps=100)
        hidden_size = 16
        actor_fn = (lambda: torch.nn.Sequential(
            torch.nn.Linear(in_features=env.state_space['shape'][0], out_features=hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=hidden_size, out_features=env.action_space['num_values'])
        ))
        critic_fn = (lambda: torch.nn.Sequential(
            torch.nn.Linear(in_features=env.state_space['shape'][0], out_features=hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=hidden_size, out_features=1)
        ))
        agent = AC(
            state_space=env.state_space, action_space=env.action_space,
            actor_fn=actor_fn, actor_learning_rate=1e-3,
            critic_fn=critic_fn, critic_learning_rate=1e-3
        )
        experiment.train(agent=agent, env=env, num_episodes=10)
        experiment.evaluate(agent=agent, env=env, num_episodes=10)
        agent.close()
        env.close()

    def test_pendulum(self):
        env = OpenAIGym(level='Pendulum', max_timesteps=100)
        hidden_size = 16
        actor_fn = (lambda: torch.nn.Sequential(
            torch.nn.Linear(in_features=env.state_space['shape'][0], out_features=hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=hidden_size, out_features=env.action_space['shape'][0])
        ))
        critic_fn = (lambda: torch.nn.Sequential(
            torch.nn.Linear(in_features=env.state_space['shape'][0], out_features=hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=hidden_size, out_features=1)
        ))
        agent = AC(
            state_space=env.state_space, action_space=env.action_space,
            actor_fn=actor_fn, actor_learning_rate=1e-3,
            critic_fn=critic_fn, critic_learning_rate=1e-3,
            discount=0.95, compute_advantage=True, normalize_returns=True
        )
        experiment.train(agent=agent, env=env, num_episodes=10)
        experiment.evaluate(agent=agent, env=env, num_episodes=10)
        agent.close()
        env.close()
