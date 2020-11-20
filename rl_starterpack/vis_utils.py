import gym
import glob
import io
import base64
import altair as alt
from IPython.display import HTML
import numpy as np
import pandas as pd
from PIL import Image


def show_episode_as_gif(ipython_display, agent, env, duration=1):
    frames = list()
    state = env.reset()
    terminal = 0
    while terminal == 0:
        frames.append(Image.fromarray(env.level.render(mode="rgb_array")))
        action = agent.act(state, evaluation=True)
        state, reward, terminal = env.step(action)
    with io.BytesIO() as stream:
        frames[0].save(
            stream, format='GIF', save_all=True, append_images=frames[1:], loop=1000, optimize=True, duration=duration
        )
        encoded = base64.b64encode(stream.getvalue()).decode('ascii')
    ipython_display.display(HTML(f'<img src="data:image/gif;base64,{encoded}" />'))


def draw_returns_chart(returns, smoothing_window=10):
    returns_df = pd.DataFrame({'raw_returns': returns, 'episode': list(range(len(returns)))})
    returns_df['smooth_returns'] = returns_df['raw_returns'].rolling(smoothing_window).mean()
    returns_long = returns_df.melt(id_vars='episode', value_vars=['raw_returns', 'smooth_returns'], var_name='type',
                                   value_name='episode_return')
    return alt.Chart(returns_long).mark_line().encode(
        x='episode',
        y='episode_return',
        color='type'
    )


def draw_loss_chart(losses, smoothing_window=10):
    losses = np.asarray(losses)
    if losses.shape[0] > 2500:
        # altair requires < 5,000 rows, so average based on multiple of 2,500
        # (since we combine raw and smoothed losses, so 2x 2,500)
        window = losses.shape[0] // 2500
        divisible = losses[:window * 2499].reshape((2499, window)).mean(axis=1)
        remainder = losses[window * 2499:].mean(axis=0, keepdims=True)
        losses = np.concatenate([divisible, remainder], axis=0)
    else:
        window = 1
    losses_df = pd.DataFrame({'raw_losses': losses, 'time': [window * n for n in range(losses.shape[0])]})
    losses_df['smooth_losses'] = losses_df['raw_losses'].rolling(smoothing_window).mean()
    losses_long = losses_df.melt(id_vars='time', value_vars=['raw_losses', 'smooth_losses'], var_name='type',
                                 value_name='loss')
    return alt.Chart(losses_long).mark_line().encode(
        x='time',
        y='loss',
        color='type'
    )
