import unittest

import torch.nn

from rl_starterpack import DQN, OpenAIGym, experiment


class TestDQN(unittest.TestCase):

    def test_frozenlake(self):
        env = OpenAIGym(level='FrozenLake', max_timesteps=100)
        num_states = env.state_space['num_values']
        hidden_size = 16
        network_fn = (lambda: torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=num_states, embedding_dim=hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=hidden_size, out_features=env.action_space['num_values'])
        ))
        agent = DQN(state_space=env.state_space, action_space=env.action_space, network_fn=network_fn,
                    learning_rate=1e-3)
        experiment.train(agent=agent, env=env, num_episodes=10)
        experiment.evaluate(agent=agent, env=env, num_episodes=10)
        agent.close()
        env.close()

    def test_taxi(self):
        env = OpenAIGym(level='Taxi', max_timesteps=100)
        num_states = env.state_space['num_values']
        hidden_size = 16
        network_fn = (lambda: torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=num_states, embedding_dim=hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=hidden_size, out_features=env.action_space['num_values'])
        ))
        agent = DQN(state_space=env.state_space, action_space=env.action_space, network_fn=network_fn,
                    learning_rate=1e-3, discount=0.95, exploration=0.1, target_network_update_frequency=10)
        experiment.train(agent=agent, env=env, num_episodes=10)
        experiment.evaluate(agent=agent, env=env, num_episodes=10)
        agent.close()
        env.close()

    def test_cartpole(self):
        env = OpenAIGym(level='CartPole', max_timesteps=100)
        hidden_size = 16
        network_fn = (lambda: torch.nn.Sequential(
            torch.nn.Linear(in_features=env.state_space['shape'][0], out_features=hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=hidden_size, out_features=env.action_space['num_values'])
        ))
        agent = DQN(state_space=env.state_space, action_space=env.action_space, network_fn=network_fn,
                    learning_rate=1e-3, target_network_update_frequency=10, memory=100, batch_size=16,
                    update_frequency=4)
        experiment.train(agent=agent, env=env, num_episodes=10)
        experiment.evaluate(agent=agent, env=env, num_episodes=10)
        agent.close()
        env.close()
