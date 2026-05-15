"""Server-side standalone vector-DQN training on FlappyBird-v0.

Mirrors the modified `run_dqn` from DQN.ipynb cell 11 exactly so we can verify
the *_best.pth saving works end-to-end before the user does it on Colab.

Run from /root/DQN:
    python3 scripts_server/train_vector_flappy.py
"""

from typing import Optional
import os

import numpy as np
import torch as th
import gymnasium as gym
import flappy_bird_gymnasium  # noqa: F401
from gymnasium import spaces

from DQN import (
    ReplayBuffer,
    collect_one_step,
    linear_schedule,
    QNetwork,
)
from DQN.evaluation import evaluate_policy


th.set_num_threads(4)  # cap so we don't fight the parallel visual-DQN GPU job


def dqn_update(q_net, q_target_net, optimizer, replay_buffer, batch_size, gamma):
    replay_data = replay_buffer.sample(batch_size).to_torch()
    with th.no_grad():
        next_q_values, _ = q_target_net(replay_data.next_observations).max(dim=1)
        should_bootstrap = th.logical_not(replay_data.terminateds)
        td_target = replay_data.rewards + gamma * next_q_values * should_bootstrap
    q_values = q_net(replay_data.observations)
    current_q_values = th.gather(q_values, dim=1, index=replay_data.actions).squeeze(dim=1)
    loss = ((current_q_values - td_target) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def run_dqn(
    env_id: str = "FlappyBird-v0",
    replay_buffer_size: int = 100_000,
    target_network_update_interval: int = 250,
    learning_starts: int = 10_000,
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.03,
    exploration_fraction: float = 0.1,
    n_timesteps: int = 500_000,
    update_interval: int = 4,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    gamma: float = 0.98,
    n_hidden_units: int = 256,
    n_eval_episodes: int = 5,
    evaluation_interval: int = 50_000,
    eval_exploration_rate: float = 0.0,
    seed: int = 2026,
    eval_render_mode: Optional[str] = None,
) -> QNetwork:
    np.random.seed(seed)
    th.manual_seed(seed)

    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./logs/checkpoint/", exist_ok=True)

    env = gym.make(env_id)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    assert isinstance(env.observation_space, spaces.Box)
    assert isinstance(env.action_space, spaces.Discrete)
    env.action_space.seed(seed)

    eval_env = gym.make(env_id, render_mode=eval_render_mode)
    eval_env = gym.wrappers.FlattenObservation(eval_env)
    eval_env.reset(seed=seed)
    eval_env.action_space.seed(seed)

    q_net = QNetwork(env.observation_space, env.action_space, n_hidden_units=n_hidden_units)
    q_target_net = QNetwork(env.observation_space, env.action_space, n_hidden_units=n_hidden_units)
    q_target_net.load_state_dict(q_net.state_dict())

    if env.observation_space.dtype == np.float64:
        q_net.double()
        q_target_net.double()

    optimizer = th.optim.Adam(q_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(replay_buffer_size, env.observation_space, env.action_space)
    obs, _ = env.reset(seed=seed)

    best_eval_return = -float("inf")
    best_checkpoint_path = f"./logs/checkpoint/q_net_checkpoint_{env_id}_best.pth"

    for current_step in range(1, n_timesteps + 1):
        exploration_rate = linear_schedule(
            exploration_initial_eps, exploration_final_eps, current_step,
            int(exploration_fraction * n_timesteps),
        )
        obs = collect_one_step(env, q_net, replay_buffer, obs, exploration_rate=exploration_rate, verbose=0)
        if current_step % target_network_update_interval == 0:
            q_target_net.load_state_dict(q_net.state_dict())
        if current_step % update_interval == 0 and current_step > learning_starts:
            dqn_update(q_net, q_target_net, optimizer, replay_buffer, batch_size, gamma=gamma)
        if current_step % evaluation_interval == 0:
            print(f"\nEvaluation at step {current_step}:")
            episode_returns = evaluate_policy(
                eval_env, q_net, n_eval_episodes, eval_exploration_rate=eval_exploration_rate
            )
            th.save(q_net.state_dict(), f"./logs/checkpoint/q_net_checkpoint_{env_id}_{current_step}.pth")
            mean_return = float(np.mean(episode_returns))
            if mean_return > best_eval_return:
                best_eval_return = mean_return
                th.save(q_net.state_dict(), best_checkpoint_path)
                print(f"New best mean eval return: {mean_return:.2f} -> {best_checkpoint_path}")

    if best_eval_return > -float("inf"):
        print(f"\nBest eval return during training: {best_eval_return:.2f} ({best_checkpoint_path})")
    return q_net


if __name__ == "__main__":
    run_dqn()
