from functools import partial
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm


def gaussian_mean_sampler(reward_mean: float = 0, reward_std: float = 1, size: int = 1):
    return np.random.normal(loc=reward_mean, scale=reward_std, size=size)


def gaussian_reward_sampler(mu: float, reward_std: float = 1, size: int = 1):
    reward = np.random.normal(loc=mu, scale=reward_std, size=size)
    if size == 1:
        return reward[0]
    return reward


class MultiArmedBandit:
    def __init__(
        self,
        num_arms: int = 10,
        arm_mean_sampler: callable = gaussian_mean_sampler,
        arm_reward_sampler: callable = gaussian_reward_sampler,
    ) -> None:
        self.seed = np.random.choice(1000000)
        self.arm_mean_sampler = arm_mean_sampler
        self.arm_reward_sampler = arm_reward_sampler
        self.num_arms = num_arms

        np.random.seed(self.seed)
        self.mean_rewards = arm_mean_sampler(size=num_arms)

    def draw_arm(self, idx: int) -> float:
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
    def __init__(
        self, multi_armed_bandit: MultiArmedBandit, policy: callable, **parameters
    ) -> None:
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
        chosen_arm = self.policy(self.expected_rewards, self.num_draws)
        reward = self.multi_armed_bandit.draw_arm(chosen_arm)

        self.total_draws += 1
        self.observed_draws.append(chosen_arm)
        self.num_draws[chosen_arm] += 1
        self.observed_rewards.append(reward)
        self.expected_rewards[chosen_arm] += (
            1.0 / (self.num_draws[chosen_arm] + 1)
        ) * (reward - self.expected_rewards[chosen_arm])

        return chosen_arm, reward

    def run(self, num_steps: int) -> None:
        for i in range(num_steps):
            _ = self.step()

    def reset(self) -> None:
        self._init_fields()

    def visualise_results(self, kind="line") -> None:
        _, ax = plt.subplots(figsize=(15, 10), nrows=3, ncols=1)

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

        sns.violinplot(
            x=self.observed_draws, y=self.observed_rewards, width=0.5, ax=ax[1]
        )
        ax[1].set_ylabel("observed reward")
        ax[1].set_xlabel("arm")

        arms, counts = np.unique(self.observed_draws, return_counts=True)
        sns.scatterplot(x=arms, y=counts, ax=ax[2])
        ax[2].set_ylim(0, len(self.observed_rewards))
        ax[2].set_ylabel("number of draws")
        ax[2].set_xlabel("arm")

        plt.show()


def average_runs(
    policy: callable,
    timesteps: int = 1000,
    repeats: int = 200,
    **parameters,
) -> None:
    for k, v in parameters.items():
        if not isinstance(v, (list, tuple)):
            parameters[k] = [v]

    parameters = ParameterGrid(parameters)
    results = []
    mean_rewards = []
    optimal_arm_draws = []

    for param_set in parameters:
        param_set_string = ", ".join(f"{k}={v}" for k, v in param_set.items())
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

    _, ax = plt.subplots(figsize=(15, 7), nrows=2, ncols=1)

    results = pd.concat(results)
    sns.lineplot(data=results, x="timestep", y="reward", hue="params", ax=ax[0])

    mean_rewards = pd.DataFrame(
        mean_rewards, columns=["iteration", "mean reward", "params"]
    )
    sns.violinplot(data=mean_rewards, x="params", y="mean reward", ax=ax[1])
    # optimal_arm_draws = pd.DataFrame(
    #     optimal_arm_draws, columns=["iteration", "optimal draws", "params"]
    # )
    # sns.violinplot(data=optimal_arm_draws, x="params", y="optimal draws", ax=ax[2])
    plt.show()
