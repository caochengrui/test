"""
Tuned Deep Q-Network trainer for visual (pixel-based) Flappy Bird.

Why a dedicated trainer instead of the generic notebook loop?
The original "visual DQN" reused hyperparameters and a training loop tuned for
*vector* observations, which is why it underperformed the vector DQN by a wide
margin on Flappy Bird. Learning a control policy from raw pixels needs a few
ingredients the vector loop did not have:

  * Double DQN target           -- removes the systematic Q-value overestimation
                                   that destabilises pixel-based training.
  * Huber (smooth L1) loss      -- robust to the large TD errors common early on.
  * Gradient-norm clipping      -- prevents occasional huge updates.
  * Reward clipping             -- keeps the TD target scale consistent.
  * A real warmup + train_freq  -- one gradient step every ``train_freq`` env
                                   steps, after a ``learning_starts`` warmup,
                                   instead of an update on (almost) every step.
  * Long epsilon annealing      -- exploration decays over a large fraction of a
                                   much longer training budget.
  * uint8 replay + GPU-side
    normalisation               -- so a large image replay buffer actually fits.

The environment side lives in :func:`DQN.wrappers.make_flappy_env`, which feeds
the native RGB ``FlappyBird-rgb-v0`` screen through grayscale -> resize ->
channels-first frame stacking.

Run from the command line for SSH debugging::

    python -m DQN.train_flappy --smoke          # ~30s pipeline sanity check
    python -m DQN.train_flappy                  # full run with defaults
    python -m DQN.train_flappy --total-timesteps 500000 --device cuda
"""

import argparse
import os
import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch as th
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces

from DQN.collect_data import epsilon_greedy_action_selection, linear_schedule
from DQN.q_network import CNNQNetwork
from DQN.replay_buffer import ReplayBuffer, TorchReplayBufferSamples
from DQN.wrappers import make_flappy_env


def get_device(device: str = "auto") -> th.device:
    """Resolve a torch device, supporting CUDA, Apple MPS, and CPU.

    ``"auto"`` prefers CUDA, then MPS (Apple Silicon), then CPU. This means the
    same code can be smoke-tested on an M1 Mac and trained for real on a GPU
    server without changes.
    """
    if device != "auto":
        return th.device(device)
    if th.cuda.is_available():
        return th.device("cuda")
    mps = getattr(th.backends, "mps", None)
    if mps is not None and mps.is_available():
        return th.device("mps")
    return th.device("cpu")


def compute_td_loss(
    q_net: CNNQNetwork,
    target_net: CNNQNetwork,
    batch: TorchReplayBufferSamples,
    gamma: float,
    double_dqn: bool,
) -> th.Tensor:
    """
    Compute the (Huber) TD loss for one minibatch.

    With ``double_dqn=True`` the next action is chosen by the online network and
    *evaluated* by the target network, which avoids the maximisation bias of
    vanilla DQN. Time-limit truncations are not stored as terminal, so the
    bootstrap mask only zeroes out genuine episode terminations.
    """
    observations = batch.observations
    next_observations = batch.next_observations
    actions = batch.actions          # (batch, 1) int64
    rewards = batch.rewards          # (batch,) float32
    terminateds = batch.terminateds  # (batch,) bool

    with th.no_grad():
        if double_dqn:
            next_actions = q_net(next_observations).argmax(dim=1, keepdim=True)
            next_q_values = target_net(next_observations).gather(1, next_actions).squeeze(1)
        else:
            next_q_values = target_net(next_observations).max(dim=1).values
        # Zero the bootstrap term for terminal transitions.
        should_bootstrap = (~terminateds).float()
        td_target = rewards + gamma * next_q_values * should_bootstrap

    current_q_values = q_net(observations).gather(1, actions).squeeze(1)
    return F.smooth_l1_loss(current_q_values, td_target)


@th.no_grad()
def evaluate(
    eval_env: gym.Env,
    q_net: CNNQNetwork,
    n_episodes: int,
    device: th.device,
    epsilon: float = 0.0,
) -> Tuple[float, float]:
    """Run ``n_episodes`` greedy episodes; return ``(mean_return, std_return)``.

    The env is neither created nor closed here so the caller can reuse it.
    """
    was_training = q_net.training
    q_net.eval()
    returns = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = epsilon_greedy_action_selection(
                q_net, obs, epsilon, eval_env.action_space, device=device
            )
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
        returns.append(total_reward)
    if was_training:
        q_net.train()
    return float(np.mean(returns)), float(np.std(returns))


def train(
    env_id: str = "FlappyBird-rgb-v0",
    total_timesteps: int = 2_000_000,
    buffer_size: int = 100_000,
    learning_starts: int = 20_000,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    train_freq: int = 4,
    target_update_interval: int = 2_000,
    exploration_fraction: float = 0.1,
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.01,
    max_grad_norm: float = 10.0,
    double_dqn: bool = True,
    frame_skip: int = 2,
    frame_stack: int = 4,
    grayscale: bool = True,
    resize_shape: Tuple[int, int] = (84, 84),
    clip_reward: bool = True,
    n_hidden_units: int = 512,
    eval_freq: int = 100_000,
    n_eval_episodes: int = 5,
    log_interval: int = 2_000,
    checkpoint_dir: str = "logs/checkpoint",
    save_path: str = "logs/checkpoint/visual_dqn_flappy.pt",
    seed: int = 2026,
    device: str = "auto",
    cpu_threads: int = 8,
) -> CNNQNetwork:
    """
    Train a pixel-based DQN agent on Flappy Bird and return the online network.

    All time-based knobs (``learning_starts``, ``target_update_interval``,
    ``eval_freq``, ``log_interval``) are counted in **environment steps**.

    Key tuning knobs when iterating on a GPU box:

    * ``total_timesteps`` -- the dominant time/quality trade-off. The default
      1M agent steps (2M game frames at ``frame_skip=2``) was verified on an
      L4 GPU to take a pixel agent from a ~9 baseline to a ~130 mean return.
    * ``frame_skip``      -- 2 (default, verified) repeats each action over two
      game frames: a good balance of control granularity and sample coverage.
      1 keeps the finest control; 4 trains faster but makes "flap" coarse.
    * ``buffer_size``     -- the dominant RAM cost (uint8 image stacks are
      stored twice, as obs and next_obs). 100k ~= 5.6 GB at (4, 84, 84);
      raise it if RAM allows, lower it on a memory-constrained box like Colab.

    :param cpu_threads: Cap on CPU worker threads (PyTorch + OpenCV). The heavy
        compute here is on the GPU; letting PyTorch/OpenCV spawn one thread per
        core only adds overhead and, worse, lets a co-running process starve
        this loop's single-threaded env stepping. A small cap is faster *and*
        plays nicely alongside other jobs on a many-core box.
    :return: The trained online ``CNNQNetwork`` (on the resolved device).
    """
    th_device = get_device(device)
    # Keep the CPU-side work (env stepping, OpenCV preprocessing, host->device
    # copies) from being drowned out by oversized thread pools.
    n_threads = max(1, min(cpu_threads, os.cpu_count() or cpu_threads))
    th.set_num_threads(n_threads)
    cv2.setNumThreads(1)  # the 84x84 grayscale/resize ops are too small to thread
    print(f"[train] device: {th_device} | cpu threads: {n_threads}")

    np.random.seed(seed)
    th.manual_seed(seed)
    if th_device.type == "cuda":
        th.cuda.manual_seed_all(seed)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # --- Environments -------------------------------------------------------
    env = make_flappy_env(
        env_id,
        frame_skip=frame_skip,
        grayscale=grayscale,
        resize_shape=resize_shape,
        frame_stack=frame_stack,
        render_mode=None,
        record_episode_stats=True,
        seed=seed,
    )
    assert isinstance(env.observation_space, spaces.Box)
    assert isinstance(env.action_space, spaces.Discrete)
    print(f"[train] observation space: {env.observation_space.shape} "
          f"{env.observation_space.dtype} | action space: {env.action_space.n}")

    eval_env: Optional[gym.Env] = None
    if eval_freq and n_eval_episodes > 0:
        eval_env = make_flappy_env(
            env_id,
            frame_skip=frame_skip,
            grayscale=grayscale,
            resize_shape=resize_shape,
            frame_stack=frame_stack,
            render_mode=None,
            record_episode_stats=False,
            seed=seed + 1,
        )
        eval_env.reset(seed=seed + 1)

    # --- Networks & optimiser ----------------------------------------------
    q_net = CNNQNetwork(env.observation_space, env.action_space, n_hidden_units).to(th_device)
    target_net = CNNQNetwork(env.observation_space, env.action_space, n_hidden_units).to(th_device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()  # the target network is only ever copied into, never trained.

    optimizer = th.optim.Adam(q_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_size, env.observation_space, env.action_space)

    n_params = sum(p.numel() for p in q_net.parameters())
    obs_bytes = int(np.prod(env.observation_space.shape))
    buffer_gb = 2 * buffer_size * obs_bytes / 1e9  # obs + next_obs
    print(f"[train] CNN parameters: {n_params:,} | "
          f"replay buffer ~{buffer_gb:.2f} GB ({buffer_size:,} transitions)")

    exploration_decay_steps = max(1, int(exploration_fraction * total_timesteps))
    recent_returns: deque = deque(maxlen=100)
    recent_lengths: deque = deque(maxlen=100)
    best_eval_return = -float("inf")
    n_updates = 0
    start_time = time.time()

    obs, _ = env.reset(seed=seed)
    for step in range(1, total_timesteps + 1):
        # --- Collect one transition with an epsilon-greedy policy -----------
        epsilon = linear_schedule(
            exploration_initial_eps, exploration_final_eps, step, exploration_decay_steps
        )
        action = epsilon_greedy_action_selection(
            q_net, obs, epsilon, env.action_space, device=th_device
        )
        next_obs, reward, terminated, truncated, info = env.step(action)
        stored_reward = float(np.clip(reward, -1.0, 1.0)) if clip_reward else float(reward)
        replay_buffer.store_transition(obs, next_obs, action, stored_reward, terminated)
        obs = next_obs

        if terminated or truncated:
            if "episode" in info:
                recent_returns.append(float(np.asarray(info["episode"]["r"]).item()))
                recent_lengths.append(float(np.asarray(info["episode"]["l"]).item()))
            obs, _ = env.reset()

        # --- Optimise the online network -----------------------------------
        if step > learning_starts and step % train_freq == 0:
            batch = replay_buffer.sample(batch_size).to_torch(th_device)
            loss = compute_td_loss(q_net, target_net, batch, gamma, double_dqn)
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(q_net.parameters(), max_grad_norm)
            optimizer.step()
            n_updates += 1

        # --- Hard-update the target network --------------------------------
        if step % target_update_interval == 0:
            target_net.load_state_dict(q_net.state_dict())

        # --- Logging -------------------------------------------------------
        if log_interval and step % log_interval == 0:
            elapsed = time.time() - start_time
            sps = step / elapsed if elapsed > 0 else 0.0
            mean_ret = np.mean(recent_returns) if recent_returns else float("nan")
            mean_len = np.mean(recent_lengths) if recent_lengths else float("nan")
            print(
                f"[train] step {step:>9,}/{total_timesteps:,} | eps {epsilon:.3f} | "
                f"ep_return {mean_ret:7.2f} | ep_len {mean_len:6.0f} | "
                f"updates {n_updates:>8,} | {sps:5.0f} step/s"
            )

        # --- Periodic evaluation + checkpoint ------------------------------
        if eval_env is not None and eval_freq and step % eval_freq == 0 and step > learning_starts:
            mean_return, std_return = evaluate(eval_env, q_net, n_eval_episodes, th_device)
            print(f"[eval ] step {step:,} | return {mean_return:.2f} +/- {std_return:.2f}")
            ckpt = os.path.join(checkpoint_dir, f"visual_dqn_flappy_{step}.pt")
            th.save(q_net.state_dict(), ckpt)
            if mean_return > best_eval_return:
                best_eval_return = mean_return
                th.save(q_net.state_dict(), os.path.join(checkpoint_dir, "visual_dqn_flappy_best.pt"))
                print(f"[eval ] new best ({mean_return:.2f}) -> visual_dqn_flappy_best.pt")

    env.close()
    if eval_env is not None:
        eval_env.close()

    th.save(q_net.state_dict(), save_path)
    print(f"[train] done. final weights saved to {save_path}")
    if best_eval_return > -float("inf"):
        print(f"[train] best eval return during training: {best_eval_return:.2f}")
    return q_net


def _write_mp4(frames: List[np.ndarray], path: str, fps: int) -> bool:
    """Write a list of RGB ``(H, W, 3)`` uint8 frames to an MP4 file via OpenCV.

    OpenCV is already a hard dependency, so this avoids relying on ``moviepy``
    (an optional ``gymnasium[other]`` extra that is easy to miss on a server).
    """
    if not frames:
        print(f"[video] no frames captured for {path}")
        return False
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return True


def record_video(
    q_net: CNNQNetwork,
    env_id: str = "FlappyBird-rgb-v0",
    video_folder: str = "logs/videos",
    name_prefix: str = "visual_dqn_flappy",
    n_episodes: int = 3,
    device: str = "auto",
    frame_skip: int = 1,
    frame_stack: int = 4,
    grayscale: bool = True,
    resize_shape: Tuple[int, int] = (84, 84),
    epsilon: float = 0.0,
    fps: int = 30,
) -> str:
    """
    Roll out a trained agent and save MP4 videos of the **real game screen**.

    The agent acts on the preprocessed (grayscale / resized / stacked)
    observations, but each frame of the saved video is the full-resolution RGB
    screen obtained from ``env.render()`` — so the video shows the actual game,
    not the 84x84 stack the network sees. ``render()`` propagates through the
    observation wrappers down to the base ``FlappyBird-rgb-v0`` env, which is
    created here with ``render_mode="rgb_array"``.

    :return: The folder the videos were written to.
    """
    th_device = get_device(device)
    os.makedirs(video_folder, exist_ok=True)

    env = make_flappy_env(
        env_id,
        frame_skip=frame_skip,
        grayscale=grayscale,
        resize_shape=resize_shape,
        frame_stack=frame_stack,
        render_mode="rgb_array",
        record_episode_stats=False,
    )

    q_net = q_net.to(th_device)
    was_training = q_net.training
    q_net.eval()
    returns = []
    with th.no_grad():
        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            frames: List[np.ndarray] = []
            while not done:
                rendered = env.render()
                if rendered is not None:
                    # pygame's surfarray is (W, H, 3); transpose to image-style
                    # (H, W, 3) so the saved video is correctly oriented.
                    frames.append(np.transpose(np.asarray(rendered), (1, 0, 2)))
                action = epsilon_greedy_action_selection(
                    q_net, obs, epsilon, env.action_space, device=th_device
                )
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                done = terminated or truncated
            returns.append(total_reward)
            _write_mp4(frames, os.path.join(video_folder, f"{name_prefix}-episode-{episode}.mp4"), fps)
    env.close()
    if was_training:
        q_net.train()

    print(f"[video] saved {n_episodes} episode(s) to '{video_folder}/' (prefix '{name_prefix}')")
    print(f"[video] return {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    return video_folder


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train visual DQN on Flappy Bird.")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--target-update-interval", type=int, default=2_000)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--no-double-dqn", action="store_true", help="Disable Double DQN.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--cpu-threads", type=int, default=8, help="Cap on PyTorch CPU threads.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run to verify the pipeline end-to-end (no real learning).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    kwargs = dict(
        total_timesteps=args.total_timesteps,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        learning_rate=args.learning_rate,
        frame_skip=args.frame_skip,
        frame_stack=args.frame_stack,
        target_update_interval=args.target_update_interval,
        eval_freq=args.eval_freq,
        double_dqn=not args.no_double_dqn,
        device=args.device,
        cpu_threads=args.cpu_threads,
        seed=args.seed,
    )
    if args.smoke:
        print("[smoke] running a tiny pipeline sanity check")
        kwargs.update(
            total_timesteps=3_000,
            buffer_size=3_000,
            learning_starts=500,
            target_update_interval=500,
            eval_freq=2_000,
            n_eval_episodes=2,
            log_interval=500,
        )
    train(**kwargs)


if __name__ == "__main__":
    main()
