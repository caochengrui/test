"""Server-side equivalent of the notebook's fair-comparison cell.

Runs both best checkpoints over the same 30 reset-seeded episodes (epsilon=0)
and prints the side-by-side table. Exactly mirrors the logic added at the end
of DQN.ipynb so the user can sanity-check the eval result before re-running
the same cell on Colab.
"""

import os
from typing import List, Union

import numpy as np
import torch as th
import gymnasium as gym
import flappy_bird_gymnasium  # noqa: F401

from DQN import QNetwork, CNNQNetwork, make_flappy_env
from DQN.collect_data import epsilon_greedy_action_selection
from DQN.train_flappy import get_device


def eval_with_seeds(
    env: gym.Env,
    q_net: th.nn.Module,
    seeds: List[int],
    device: Union[str, th.device] = "cpu",
    epsilon: float = 0.0,
    max_steps: int = 20_000,
) -> np.ndarray:
    was_training = q_net.training
    q_net.eval()
    returns = []
    with th.no_grad():
        for s in seeds:
            obs, _ = env.reset(seed=int(s))
            total = 0.0
            done = False
            steps = 0
            while not done and steps < max_steps:
                action = epsilon_greedy_action_selection(
                    q_net, obs, epsilon, env.action_space, device=device
                )
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                done = terminated or truncated
                steps += 1
            returns.append(total)
    if was_training:
        q_net.train()
    return np.asarray(returns, dtype=np.float64)


def _row(name: str, r: np.ndarray) -> str:
    return (
        f"{name:<11} | n={len(r):>2d} | "
        f"mean {r.mean():>8.2f} | median {np.median(r):>8.2f} | "
        f"std {r.std():>8.2f} | min {r.min():>7.2f} | max {r.max():>7.2f}"
    )


def main() -> None:
    N_EVAL_EPISODES = 30
    EVAL_SEEDS = list(range(10_000, 10_000 + N_EVAL_EPISODES))

    VECTOR_CKPT = "./logs/checkpoint/q_net_checkpoint_FlappyBird-v0_best.pth"
    VISUAL_CKPT = "./logs/checkpoint/visual_dqn_flappy_best.pt"

    for ckpt in (VECTOR_CKPT, VISUAL_CKPT):
        assert os.path.exists(ckpt), f"Missing checkpoint {ckpt}"

    # Vector
    vector_env = gym.make("FlappyBird-v0")
    vector_env = gym.wrappers.FlattenObservation(vector_env)
    vector_q_net = QNetwork(
        vector_env.observation_space, vector_env.action_space, n_hidden_units=256
    )
    vector_q_net.double()
    vector_q_net.load_state_dict(th.load(VECTOR_CKPT, map_location="cpu"))
    print(f"[compare] evaluating Vector DQN on {N_EVAL_EPISODES} episodes ...")
    vector_returns = eval_with_seeds(vector_env, vector_q_net, EVAL_SEEDS, device="cpu")
    vector_env.close()

    # Visual
    visual_device = get_device("auto")
    visual_env = make_flappy_env(
        "FlappyBird-rgb-v0",
        frame_skip=2, frame_stack=4, grayscale=True,
        record_episode_stats=False,
    )
    visual_q_net = CNNQNetwork(
        visual_env.observation_space, visual_env.action_space, n_hidden_units=512,
    ).to(visual_device)
    visual_q_net.load_state_dict(th.load(VISUAL_CKPT, map_location=visual_device))
    print(f"[compare] evaluating Visual DQN on {N_EVAL_EPISODES} episodes ({visual_device}) ...")
    visual_returns = eval_with_seeds(
        visual_env, visual_q_net, EVAL_SEEDS, device=visual_device,
    )
    visual_env.close()

    print()
    print(f"Fair comparison on {N_EVAL_EPISODES} episodes "
          "(epsilon=0, same eval seeds for both agents, best-by-eval checkpoints):")
    print("-" * 99)
    print(_row("Vector DQN", vector_returns))
    print(_row("Visual DQN", visual_returns))
    print("-" * 99)
    diff = visual_returns.mean() - vector_returns.mean()
    if vector_returns.mean() > 1e-9:
        ratio = visual_returns.mean() / vector_returns.mean()
        print(f"Visual - Vector mean diff: {diff:+.2f}  ({ratio:.2f}x)")
    else:
        print(f"Visual - Vector mean diff: {diff:+.2f}")
    print(
        "Note: Vector DQN sees exact distances/velocities (12-D float vector), "
        "Visual DQN must learn from 84x84 grayscale pixels."
    )

    np.savez(
        "./logs/comparison_returns.npz",
        vector_returns=vector_returns,
        visual_returns=visual_returns,
        eval_seeds=np.asarray(EVAL_SEEDS),
    )
    print("[compare] returns saved to logs/comparison_returns.npz")


if __name__ == "__main__":
    main()
