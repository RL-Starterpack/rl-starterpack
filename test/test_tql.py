import unittest

from rl_starterpack import OpenAIGym, TQL, experiment


class TestTQL(unittest.TestCase):

    def test_frozenlake(self):
        env = OpenAIGym(level='FrozenLake', max_timesteps=100)
        agent = TQL(state_space=env.state_space, action_space=env.action_space, learning_rate=0.3)
        experiment.train(agent=agent, env=env, num_episodes=10)
        experiment.evaluate(agent=agent, env=env, num_episodes=10)
        agent.close()
        env.close()

    def test_taxi(self):
        env = OpenAIGym(level='Taxi', max_timesteps=100)
        agent = TQL(
            state_space=env.state_space, action_space=env.action_space,
            learning_rate=0.3, discount=0.95, exploration=0.1
        )
        experiment.train(agent=agent, env=env, num_episodes=10)
        experiment.evaluate(agent=agent, env=env, num_episodes=10)
        agent.close()
        env.close()
