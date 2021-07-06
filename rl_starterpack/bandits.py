from functools import partial
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm


def gaussian_mean_sampler(reward_mean: float = 0, reward_std: float = 1, size: int = 1):
    """
    Sample a mean from a Normal distribution with given mean and std.
    """
    return np.random.normal(loc=reward_mean, scale=reward_std, size=size)


def gaussian_reward_sampler(mu: float, reward_std: float = 1, size: int = 1):
    """
    Sample a reward for a Normal distribution with given mean and std
    """
    reward = np.random.normal(loc=mu, scale=reward_std, size=size)
    if size == 1:
        return reward[0]
    return reward


class MultiArmedBandit:
    """
    Object that defines the multi-armed bandit environment. The environment
    has `num_arms` arms, each with a mean reward sampled from
    `arm_mean_sampler`, and, each producing a reward, upon pulling by
    sampling from `arm_reward_sampler`.
    """

    def __init__(
        self,
        num_arms: int = 10,
        arm_mean_sampler: callable = gaussian_mean_sampler,
        arm_reward_sampler: callable = gaussian_reward_sampler,
    ) -> None:
        self.arm_mean_sampler = arm_mean_sampler
        self.arm_reward_sampler = arm_reward_sampler
        self.num_arms = num_arms

        self.mean_rewards = arm_mean_sampler(size=num_arms)

    def draw_arm(self, idx: int) -> float:
        """Draw arm `idx` and return the observed reward.

        Args:
            idx (int): index of arm to draw

        Returns:
            float: observed reward sampled with `arm_reward_sampler`
        """

        mu = self.mean_rewards[idx]
        return self.arm_reward_sampler(mu)

    def visualise(self) -> None:
        dfs = []
        for arm, mu in enumerate(self.mean_rewards):
            dummies = self.arm_reward_sampler(mu, size=1000)
            dfs.append(pd.DataFrame({"Arm": arm, "Reward Distribution": dummies}))

        dfs = pd.concat(dfs)
        _, ax = plt.subplots(figsize=(10, 5))
        sns.violinplot(data=dfs, x="Arm", y="Reward Distribution", width=0.3, ax=ax)
        plt.show()


class BanditExperiment:
    """Experiment class to perform multiple draws on a MultiArmedBandit. A policy
    function needs to provided, along witht he parameters used for the policy. The
    arm for each draw is determined using the policy. This object keeps track of the
    rewards observed at each timestep and the expected reward for each arm over arm
    draws.
    """

    def __init__(
        self, multi_armed_bandit: MultiArmedBandit, policy: callable, **parameters
    ) -> None:
        """Constructor for experiment object.

        Args:
            multi_armed_bandit (MultiArmedBandit): bandit object to perform
                experiment on
            policy (callable): function that decides which arm to choose at each
                timestep given the expected rewards for each arm
        """
        self.multi_armed_bandit = multi_armed_bandit
        self.policy = policy
        self.init_rewards = parameters.get("init_rewards", 0)
        self._init_fields()

    def _init_fields(self) -> None:
        self.observed_rewards = []
        self.expected_rewards = self.init_rewards * np.ones(
            self.multi_armed_bandit.num_arms
        )
        self.observed_draws = []
        self.num_draws = np.zeros(self.multi_armed_bandit.num_arms, dtype=np.int32)
        self.total_draws = 0

    def step(self) -> Tuple[int, float]:
        """Perform one timestep by choosing an arm according to policy, updating
        the expected reward for the chosen arm, and keeping track of observed
        rewards.

        Returns:
            Tuple[int, float]: Arm chosen and observed reward in this timestep
        """
        chosen_arm = self.policy(self.expected_rewards, self.num_draws)
        assert (
            isinstance(chosen_arm, (int, np.int32, np.int64))
            and 0 <= chosen_arm < self.multi_armed_bandit.num_arms
        ), f"Please make sure the policy returns an integer in range [0, {self.multi_armed_bandit.num_arms})"
        reward = self.multi_armed_bandit.draw_arm(chosen_arm)

        self.total_draws += 1
        self.observed_draws.append(chosen_arm)
        self.num_draws[chosen_arm] += 1
        self.observed_rewards.append(reward)

        # Instead of keeping track of total observed rewards for each arm in a
        # separate array, we update the expected reward dynamically using the
        # difference between the observed and expected rewards. For details,
        # refer to Eqn. 2.3 in the book by Sutton & Barto at
        # http://incompleteideas.net/book/RLbook2020.pdf
        self.expected_rewards[chosen_arm] += (
            1.0 / (self.num_draws[chosen_arm] + 1)
        ) * (reward - self.expected_rewards[chosen_arm])

        return chosen_arm, reward

    def run(self, num_steps: int) -> None:
        """Run experiment for given timesteps

        Args:
            num_steps (int): Number of timesteps to run
        """
        for i in range(num_steps):
            _ = self.step()

    def reset(self) -> None:
        """Reset experiment data"""
        self._init_fields()

    def visualise_results(self, kind="line") -> None:
        """Plot the results of the experiment. Shows the observed rewards over
        timesteps, the distribution of observed rewards for each arm and the
        number of times each arm was drawn.

        Args:
            kind (str, optional): Plot to draw for the observed reward. One of
                ['line', 'scatter']. Defaults to "line".
        """
        fig, ax = plt.subplots(figsize=(15, 10), nrows=3, ncols=1)

        reward_df = pd.DataFrame(
            {
                "timestep": range(len(self.observed_rewards)),
                "reward_raw": self.observed_rewards,
            }
        )
        reward_df["reward"] = reward_df["reward_raw"].rolling(10).mean()

        if kind == "line":
            sns.lineplot(data=reward_df, x="timestep", y="reward", ax=ax[0])
        else:
            sns.scatterplot(
                data=reward_df, x="timestep", y="reward_raw", hue="reward_raw", ax=ax[0]
            )
        ax[0].set_title("Reward observed at each timestep")

        sns.violinplot(
            x=self.observed_draws, y=self.observed_rewards, width=0.5, ax=ax[1]
        )
        ax[1].set_ylabel("observed reward")
        ax[1].set_xlabel("arm")
        ax[1].set_title("Distribution of observed rewards for each arm")

        arms, counts = np.unique(self.observed_draws, return_counts=True)
        sns.scatterplot(x=arms, y=counts, ax=ax[2])
        ax[2].set_ylim(0, len(self.observed_rewards))
        ax[2].set_ylabel("number of draws")
        ax[2].set_xlabel("arm")
        ax[2].set_title("Number of draws of each arm")

        fig.tight_layout()
        plt.show()


def average_runs(
    policy: callable,
    timesteps: int = 1000,
    repeats: int = 200,
    **parameters,
) -> None:
    """Run experiment on separate MultiArmedBandit objects and average the results.
    Also aids in visually examining the results.

    Args:
        policy (callable): Policy function to run experiments with
        timesteps (int, optional): Timesteps to run each experiment for.
            Defaults to 1000.
        repeats (int, optional): Number of independent experiments to run.
            Defaults to 200.
    """
    for k, v in parameters.items():
        if not isinstance(v, (list, tuple)):
            parameters[k] = [v]

    parameters = ParameterGrid(parameters)
    results = []
    mean_rewards = []
    optimal_arm_draws = []

    for param_set in parameters:
        param_set_string = ", ".join(f"{k}: {v}" for k, v in param_set.items())
        observed_rewards = []
        for n in tqdm(range(repeats)):
            multi_armed_bandit = MultiArmedBandit()
            optimal_arm = multi_armed_bandit.mean_rewards.argmax()
            policy = partial(policy, **param_set)
            experiment = BanditExperiment(
                multi_armed_bandit=multi_armed_bandit, policy=policy, **param_set
            )

            experiment.run(num_steps=timesteps)
            observed_rewards.append(experiment.observed_rewards)
            optimal_arm_draws.append(
                (n, experiment.num_draws[optimal_arm], param_set_string)
            )
            mean_rewards.append(
                (n, np.array(experiment.observed_rewards).mean(), param_set_string)
            )

        observed_rewards = np.array(observed_rewards).T.flatten()
        rewards_df = pd.DataFrame(
            {
                "timestep": np.repeat(range(timesteps), repeats),
                "reward_raw": observed_rewards,
                "params": param_set_string,
            }
        )
        rewards_df["reward"] = rewards_df["reward_raw"].rolling(10).mean()
        results.append(rewards_df)

    fig, ax = plt.subplots(figsize=(15, 7), nrows=2, ncols=1)

    results = pd.concat(results)
    sns.lineplot(data=results, x="timestep", y="reward", hue="params", ax=ax[0])
    ax[0].set_title(f"Reward per timestep averaged over {repeats} experiments")

    plt.grid(True)
    optimal_arm_draws = pd.DataFrame(
        optimal_arm_draws, columns=["iteration", "optimal draws", "params"]
    )
    sns.histplot(
        data=optimal_arm_draws,
        x="optimal draws",
        hue="params",
        element="step",
        # multiple="dodge",
        ax=ax[1],
    )
    ax[1].set_title(
        f"Distribution of Optimal-Arm-Draws/{timesteps} timesteps over {repeats} experiments"
    )

    fig.tight_layout()
    plt.show()
