import unittest

from rl_starterpack import OpenAIGym, RandomAgent, experiment


class TestRandomAgent(unittest.TestCase):

    def test_cartpole(self):
        env = OpenAIGym(level='CartPole', max_timesteps=100)
        agent = RandomAgent(state_space=env.state_space, action_space=env.action_space)
        experiment.train(agent=agent, env=env, num_episodes=10)
        experiment.evaluate(agent=agent, env=env, num_episodes=10)
        agent.close()
        env.close()

    def test_pendulum(self):
        env = OpenAIGym(level='Pendulum', max_timesteps=100)
        agent = RandomAgent(state_space=env.state_space, action_space=env.action_space)
        experiment.train(agent=agent, env=env, num_episodes=10)
        experiment.evaluate(agent=agent, env=env, num_episodes=10)
        agent.close()
        env.close()
