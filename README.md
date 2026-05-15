# DQN - Deep Q-Network Implementation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/caochengrui/DQN/blob/main/DQN.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.4.0-ee4c2c.svg)](https://pytorch.org/)

A modular implementation of the Deep Q-Network (DQN) algorithm built with [Gymnasium](https://gymnasium.farama.org/) and [PyTorch](https://pytorch.org/).

This repository supports two distinct DQN paths:

- **Vector DQN** — an MLP-based `QNetwork` for low-dimensional vector observations (`CartPole-v1`, the feature-vector `FlappyBird-v0`), trained with a hand-written loop in the notebook.
- **Visual DQN** — a Nature-DQN-style `CNNQNetwork` that learns **directly from pixels**, with a tuned, batteries-included trainer (`DQN/train_flappy.py`) for `FlappyBird-rgb-v0`.

It also includes a replay buffer, epsilon-greedy data collection helpers, evaluation and video-recording helpers, the Flappy Bird visual preprocessing pipeline, and a notebook with end-to-end training examples.

---

## Table of Contents

- [Overview](#overview)
- [Algorithm](#algorithm)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Public API](#public-api)
- [Quick Start](#quick-start)
- [Visual Observation Pipeline](#visual-observation-pipeline)
- [Evaluation and Video Recording](#evaluation-and-video-recording)
- [Verified Results](#verified-results)
- [Supported Environments and Scope](#supported-environments-and-scope)
- [Acknowledgements](#acknowledgements)

---

## Overview

The codebase focuses on reusable DQN building blocks rather than a large training framework. It implements the standard DQN ingredients:

- an online Q-network
- a target network
- an experience replay buffer
- epsilon-greedy exploration
- helper utilities for evaluation and video recording

Two network backbones are provided out of the box:

- `QNetwork`: a 2-hidden-layer MLP for 1D vector observations
- `CNNQNetwork`: a Nature-DQN convolutional Q-network for channels-first image stacks

The repository is intentionally lightweight: an installable package plus [DQN.ipynb](DQN.ipynb) for experiments. The vector-DQN training loop is written out in the notebook; the visual-DQN training loop is a real, tuned module — `DQN/train_flappy.py` — which also runs as a script and includes Double DQN, Huber loss, gradient clipping, and reward clipping. Dueling DQN and prioritized replay are not included.

---

## Algorithm

For a non-terminal transition, the standard DQN target is:

$$
y_t = r_t + \gamma (1 - d_t) \max_{a'} Q_{\text{target}}(s_{t+1}, a')
$$

where:

- $r_t$ is the reward
- $\gamma$ is the discount factor
- $d_t$ indicates whether the episode terminated
- $Q_{\text{target}}$ is the target network

A typical training loop in this repository looks like:

1. collect transitions with an epsilon-greedy policy
2. store them in the replay buffer
3. sample a mini-batch
4. compute TD targets with the target network
5. update the online Q-network
6. periodically synchronize the target network
7. evaluate the policy

Implementation note: the replay buffer stores `terminated` flags, while time-limit truncations are handled by resetting the environment during collection instead of being treated as absorbing terminal states.

---

## Features

- MLP-based DQN for vector observations via `QNetwork`
- CNN-based DQN for pixel observations via `CNNQNetwork`
- tuned visual-DQN trainer (`DQN.train_flappy.train`): Double DQN, Huber loss, gradient + reward clipping, warmup, configurable `train_freq` and target-sync interval
- Flappy Bird visual preprocessing pipeline (`make_flappy_env`): action repeat, grayscale, resize, channels-first frame stacking
- replay buffer with `uint8` NumPy storage and `.to_torch()` conversion
- epsilon-greedy action selection and linear exploration schedule
- one-step collection helper for notebook-style (vector-DQN) training loops
- video recording of trained agents (`record_video`) via OpenCV
- automatic device selection (CUDA → Apple MPS → CPU)
- Google Colab notebook for interactive experimentation
- installable package via `pip`

---

## Project Structure

```text
DQN/
├── DQN/                        # Core package
│   ├── __init__.py             # Public package exports
│   ├── collect_data.py         # Epsilon-greedy action selection and rollout helpers
│   ├── evaluation.py           # Evaluation and optional video recording
│   ├── q_network.py            # MLP and CNN Q-network definitions
│   ├── replay_buffer.py        # Replay buffer and batch containers
│   ├── wrappers.py             # Flappy Bird pixel preprocessing + make_flappy_env
│   └── train_flappy.py         # Tuned visual-DQN trainer + video recording (also a CLI)
├── DQN.ipynb                   # End-to-end notebook examples
├── custom_utils.py             # Notebook helper for displaying recorded videos
├── pyproject.toml              # Packaging, dependencies, and tool configuration
└── README.md                   # Project documentation
```

---

## Installation

### Option 1: Install directly from GitHub

```bash
pip install "git+https://github.com/caochengrui/DQN.git"
```

### Option 2: Clone and install locally

```bash
git clone https://github.com/caochengrui/DQN.git
cd DQN
pip install -e .
```

### Option 3: Open the notebook in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/caochengrui/DQN/blob/main/DQN.ipynb)

### Optional setup

If you want to record evaluation videos, install `ffmpeg` first.

On Debian/Ubuntu or Google Colab:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

On macOS:

```bash
brew install ffmpeg
```

If you want to run `FlappyBird-v0`, install the environment separately:

```bash
pip install "flappy-bird-gymnasium @ git+https://github.com/araffin/flappy-bird-gymnasium@patch-1"
```

---

## Dependencies

Core runtime dependencies from `pyproject.toml`:

- Python >= 3.8
- PyTorch >= 2.4.0
- Gymnasium >= 0.29.1, < 1.1.0 with `classic-control` and `other` extras
- NumPy
- scikit-learn
- opencv-python >= 4.6.0

The repository also keeps tool configuration for `ruff`, `black`, and `mypy` in `pyproject.toml`.

---

## Public API

The package root (`from DQN import ...`) exports the building blocks:

- `QNetwork`: MLP for discrete-action environments with 1D `Box` observations
- `CNNQNetwork`: Nature-DQN CNN for channels-first `(C, H, W)` image stacks
- `ReplayBuffer`: ring-buffer replay storage
- `epsilon_greedy_action_selection`: action selection helper
- `collect_one_step`: collect one transition and store it in the replay buffer
- `linear_schedule`: linear epsilon schedule
- `SkipEnv`, `GrayscaleWrapper`, `ResizeWrapper`, `FrameStack`: Flappy Bird preprocessing wrappers
- `make_flappy_env`: factory for the `FlappyBird-rgb-v0` visual pipeline

The visual-DQN trainer is imported explicitly (not re-exported, so `python -m DQN.train_flappy` runs cleanly):

- `DQN.train_flappy.train`: the tuned pixel-DQN training loop
- `DQN.train_flappy.record_video`: roll out a trained agent and save MP4s
- `DQN.train_flappy.evaluate`, `DQN.train_flappy.get_device`

Additional helpers:

- `DQN.evaluation.evaluate_policy` (used by the vector-DQN path)
- `custom_utils.notebook_show_videos`

---

## Quick Start

### Vector observations

```python
import gymnasium as gym
import torch as th
import torch.nn.functional as F
from torch import optim

from DQN import QNetwork, ReplayBuffer, collect_one_step, linear_schedule

env = gym.make("CartPole-v1")
obs, _ = env.reset()

q_net = QNetwork(env.observation_space, env.action_space)
target_net = QNetwork(env.observation_space, env.action_space)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
replay_buffer = ReplayBuffer(
    buffer_size=100_000,
    observation_space=env.observation_space,
    action_space=env.action_space,
)

for step in range(20_000):
    epsilon = linear_schedule(1.0, 0.05, step, 10_000)
    obs = collect_one_step(env, q_net, replay_buffer, obs, exploration_rate=epsilon)

    if not replay_buffer.is_full and replay_buffer.current_idx < 32:
        continue

    batch = replay_buffer.sample(32).to_torch()

    with th.no_grad():
        next_q_values = target_net(batch.next_observations).max(dim=1).values
        td_target = batch.rewards + 0.99 * next_q_values * (~batch.terminateds)

    current_q_values = q_net(batch.observations).gather(1, batch.actions).squeeze(1)
    loss = F.mse_loss(current_q_values, td_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        target_net.load_state_dict(q_net.state_dict())
```

### Visual DQN on Flappy Bird (pixels)

The visual path ships a complete, tuned trainer — you do not write the loop yourself:

```python
from DQN.train_flappy import train, record_video

# Trains a CNN DQN directly on the FlappyBird-rgb-v0 game screen and returns
# the trained network. All time-based arguments are counted in env steps.
# This config was verified on an L4 GPU: ~9 baseline -> ~130 mean eval return.
q_net = train(
    env_id="FlappyBird-rgb-v0",
    total_timesteps=1_000_000,   # main quality/time knob; 2M game frames at frame_skip=2
    buffer_size=100_000,
    learning_starts=20_000,
    frame_skip=2,
    frame_stack=4,
    double_dqn=True,
    device="auto",               # CUDA -> Apple MPS -> CPU
)

# Save MP4s of the full-resolution game played by the trained agent.
record_video(q_net, env_id="FlappyBird-rgb-v0", video_folder="logs/videos")
```

Or run it as a script (handy for remote GPU boxes):

```bash
python -m DQN.train_flappy --smoke                       # ~30s pipeline sanity check
python -m DQN.train_flappy --total-timesteps 1500000     # full run
```

`make_flappy_env` turns the RGB game screen into a channels-first `(4, 84, 84)` `uint8` stack, which `CNNQNetwork` consumes directly (normalizing `uint8` to `[0, 1]` internally). For the full workflow and video visualization, see [DQN.ipynb](DQN.ipynb).

---

## Visual Observation Pipeline

`make_flappy_env()` builds the preprocessing pipeline for pixel-based Flappy Bird DQN. The base environment is `FlappyBird-rgb-v0`, whose observation is already the rendered RGB game screen — so no "render to pixels" wrapper is needed.

The wrappers are applied in this order:

1. `SkipEnv` repeats each action `frame_skip` times, summing rewards (only if `frame_skip > 1`; no max-pooling, since the renderer does not flicker)
2. `GrayscaleWrapper` converts RGB frames to grayscale
3. `ResizeWrapper` resizes frames to `resize_shape` (default `84 x 84`)
4. `FrameStack` stacks `frame_stack` consecutive frames into a channels-first `(frame_stack, H, W)` `uint8` array — this is what enforces the `CNNQNetwork` input contract
5. `gym.wrappers.RecordEpisodeStatistics` records episode returns and lengths

`make_flappy_env` also configures SDL for headless machines (Colab, GPU servers), so no display is required.

---

## Evaluation and Video Recording

The visual path records videos of a trained agent with `record_video`, which writes MP4s of the **full-resolution** game screen (not the 84×84 stack the network sees) using OpenCV:

```python
from DQN.train_flappy import record_video
from custom_utils import notebook_show_videos

record_video(q_net, env_id="FlappyBird-rgb-v0", video_folder="logs/videos",
             name_prefix="visual_dqn_flappy", n_episodes=3)

notebook_show_videos("logs/videos", prefix="visual_dqn_flappy")
```

The vector-DQN path uses `DQN.evaluation.evaluate_policy` instead (see [DQN.ipynb](DQN.ipynb)).

---

## Verified Results

Both DQN paths were trained and evaluated on an NVIDIA L4 GPU on the same game
(`flappy-bird-gymnasium`), then scored over 30 greedy evaluation episodes:

| Agent | Observation | Training budget | Mean | Median | Max |
|---|---|---|---|---|---|
| **Visual DQN** | 84×84 grayscale pixels, 4-frame stack | 1M steps (2M game frames) | **130.9** | **112.0** | **391.6** |
| Vector DQN | 12-D feature vector | 2M env steps | 34.7 | 27.5 | 83.5 |

The visual DQN's evaluation return climbed smoothly and monotonically
(9 → 15 → 36 → 43 → 120 → 137 at each 100k-step checkpoint). The vanilla
vector DQN peaked at ~40 then destabilised (collapsing to 4–9 before partially
recovering) — the Double DQN target, Huber loss and gradient clipping in the
visual trainer are what keep its learning curve stable.

---

## Supported Environments and Scope

This repository is designed for discrete-action Gymnasium environments.

Supported out of the box:

- vector-observation environments with 1D `spaces.Box` observations, such as `CartPole-v1`
- the feature-vector `FlappyBird-v0` (vector DQN), after installing the optional environment package
- the pixel-based `FlappyBird-rgb-v0` (visual DQN), via `make_flappy_env` + `DQN.train_flappy.train`

Important limitations:

- the action space must be `spaces.Discrete`
- `QNetwork` expects 1D vector observations
- `CNNQNetwork` expects channels-first 3D `(C, H, W)` image stacks (produced by `make_flappy_env`)
- the visual pipeline is intentionally Flappy-Bird-focused; CartPole is **not** supported as a visual task (rendering it to pixels discards the state that makes it learnable)
- there is no built-in support for continuous control
- dueling DQN and prioritized replay are not included (Double DQN is, in the visual trainer)

The CNN and wrappers can be adapted to other pixel environments, but the `make_flappy_env` factory targets `FlappyBird-rgb-v0` specifically.

---

## Acknowledgements

- This project is based on the [RLSS 2023 DQN tutorial](https://github.com/araffin/rlss23-dqn) by [Antonin Raffin](https://github.com/araffin).
- Some hyperparameter choices are inspired by the [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo).
- The optional Flappy Bird environment uses the [patch-1 fork of flappy-bird-gymnasium](https://github.com/araffin/flappy-bird-gymnasium/tree/patch-1).
