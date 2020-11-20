import numpy as np
from tqdm.auto import tqdm


def train(agent, env, num_episodes, reward_shaping_fn=None, use_pbar=True):
    """Initialize and train an agent on an environment.

    Args:
        agent (Agent): Agent.
        env (Environment): Environment.
        num_episodes (int): Number of episodes.
        reward_shaping_fn (callable[(reward, terminal, next-state) -> (reward, terminal)]):
            Optional reward shaping.
        use_pbar (bool): Whether a progress bar should be shown.

    Returns:
        np.array[float]: Episode returns.
    """
    # Training loop
    returns = list()
    episodes = range(num_episodes)
    if use_pbar:
        episodes = tqdm(episodes)
        episodes.set_postfix({'return': 'n/a'})
    for _ in episodes:
        returns.append(0.0)

        # Episode loop
        state = env.reset()
        terminal = 0
        while terminal == 0:
            action = agent.act(state=state)
            next_state, reward, terminal = env.step(action=action)
            if reward_shaping_fn is not None:
                reward, terminal = reward_shaping_fn(reward, terminal, next_state)
            agent.observe(
                state=state, action=action, reward=reward, terminal=terminal, next_state=next_state
            )
            state = next_state
            returns[-1] += reward

        # Record episode return
        if use_pbar:
            episodes.set_postfix({'return': '{:.2f}'.format(returns[-1])})

    return np.array(returns)


def evaluate(agent, env, num_episodes, use_pbar=True):
    """Evaluate an agent on an environment.

    Args:
        agent (Agent): Agent.
        env (Environment): Environment.
        num_episodes (int): Number of episodes.
        use_pbar (bool): Whether a progress bar should be shown.

    Returns:
        np.array[float]: Episode returns.
    """

    # Evaluation loop
    returns = list()
    episodes = range(num_episodes)
    if use_pbar:
        episodes = tqdm(episodes)
        episodes.set_postfix({'return': 'n/a'})
    for _ in episodes:
        returns.append(0.0)

        # Episode loop
        state = env.reset()
        terminal = 0
        while terminal == 0:
            action = agent.act(state=state, evaluation=True)
            state, reward, terminal = env.step(action=action)
            returns[-1] += reward

        # Record episode return
        if use_pbar:
            episodes.set_postfix({'return': '{:.2f}'.format(returns[-1])})

    return np.array(returns)


def multi_run_train_eval(
        agent_fn, env, num_episodes_train, num_episodes_eval=100, num_runs=10, reward_shaping_fn=None,
):
    """Train and evaluate an agent on an environment over multiple runs.
       Returns the average evaluate score across these runs.

    Args:
        agent_fn (callable[-> Agent]): Agent constructor.
        env (Environment): Environment.
        num_episodes_train (int): Number of episodes for training per run.
        num_episodes_eval (int): Number of episodes for evaluation per run.
        num_runs (int): Number of training-evaluation runs to average over.
        reward_shaping_fn (callable[(reward, terminal, next-state) -> (reward, terminal)]):
            Optional reward shaping used for training.

    Return:
        np.array[float]: Mean evaluation return for each run
    """
    # Average run loop
    run_returns = list()
    pbar = tqdm(range(num_runs))
    pbar.set_postfix({'return': 'n/a'})
    for _ in pbar:
        # Initialize agent
        agent = agent_fn()

        _ = train(agent, env, num_episodes_train, reward_shaping_fn, use_pbar=False)

        # Evaluation loop
        eval_returns = evaluate(agent, env, num_episodes_eval, use_pbar=False)

        # Close agent
        agent.close()

        # Record evaluation return
        run_returns.append(eval_returns.mean())
        pbar.set_postfix({'return': '{:.2f}'.format(run_returns[-1])})

    return np.array(run_returns)


def evaluate_render(agent, env, ipython_display, render_frequency=1, sleep=0.0):
    """Evaluate and render an agent on an environment.

    Args:
        agent (Agent): Agent.
        env (Environment): Environment.
        ipython_display (IPython.display): IPython display.
        render_frequency (int): Render frequency.
        sleep (float): Sleep slowdown in seconds.
        fn_render (env -> RGB array): Custom render function.
    """
    assert render_frequency >= 1
    assert sleep >= 0.0

    # Prepare
    timestep = 0
    state = env.reset()
    action = 'n/a'
    reward = 0.0
    episode_return = 0.0

    # Episode loop
    while True:
        if timestep % render_frequency == 0:
            env.render(ipython_display, sleep)
            print('timestep:', timestep, '|', 'reward:', reward, '|', 'return:', episode_return)
            str_state = str(state)
            if len(str_state) <= 20 and '\n' not in str_state:
                print('state:', str_state, end=' ')
            str_action = str(action)
            if len(str_action) <= 20 and '\n' not in str_action:
                print('|', 'action:', str_action, end=' ')
            print()
        action = agent.act(state=state, evaluation=True)
        state, reward, terminal = env.step(action=action)
        timestep += 1
        episode_return += reward
        if terminal > 0:
            break

    # Final render
    env.render(ipython_display, sleep)
    env.close()
    print('timestep:', timestep, '|', 'return:', episode_return, '|', 'terminal:', terminal)
