# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e .              # install the package in editable mode
ruff check .                  # lint (rules E,F,B,UP,C90,RUF; line-length 127)
black .                       # format (line-length 127)
mypy DQN custom_utils.py      # type-check

python -m DQN.train_flappy --smoke   # ~30s pipeline sanity check (no real learning)
python -m DQN.train_flappy           # full visual-DQN run on Flappy Bird
```

There is no automated test suite. The vector-DQN training loop lives in `DQN.ipynb`; the visual-DQN training loop is a real module, `DQN/train_flappy.py` (also runnable as a script). The package version is derived from git tags via `setuptools_scm` (writes `DQN/_version.py`), so a clean git checkout is required to build.

## Two distinct DQN paths

The repo deliberately keeps two separate paths; do not cross-wire them.

- **Vector DQN** — `QNetwork` (MLP) + `collect_one_step` + a hand-written loop in `DQN.ipynb`. Used for `CartPole-v1` and the *feature-vector* `FlappyBird-v0`.
- **Visual DQN** — `CNNQNetwork` + `make_flappy_env` + the `DQN.train_flappy.train` loop. Used **only** for pixel-based `FlappyBird-rgb-v0`. CartPole is intentionally *not* supported as a visual task: rendering it to pixels throws away the exact state that makes it learnable.

## Architecture

The package provides reusable DQN building blocks plus one batteries-included visual trainer. Building blocks are re-exported from `DQN/__init__.py`; the trainer is imported explicitly (`from DQN.train_flappy import train, record_video`) and is *not* re-exported, so `python -m DQN.train_flappy` runs without a double-import warning.

**Q-networks (`q_network.py`)** — `QNetwork` is a 2-hidden-layer MLP for 1D `Box` observations. `CNNQNetwork` is a Nature-DQN CNN that expects **channels-first `(C, H, W)`** image stacks only — there is no layout auto-detection; the preprocessing pipeline is the single source of truth for the observation format. Conv flatten size is computed via a dummy forward pass; weights use orthogonal init (SB3-style). `uint8` observation spaces trigger automatic `/255.0` normalization inside `forward()`, so the replay buffer can keep cheap `uint8` frames.

**Replay buffer (`replay_buffer.py`)** — `ReplayBuffer` is a NumPy ring buffer. It stores only `terminated` flags, **not** truncations — time-limit truncations are handled by resetting the env during collection, never as absorbing states. `sample()` returns a `ReplayBufferSamples` (NumPy); call `.to_torch(device)` to get a `TorchReplayBufferSamples`. It stores `observations` and `next_observations` separately, so for `(4, 84, 84)` uint8 stacks memory is ~`2 * buffer_size * 28 KB` (~5.6 GB at 100k) — the dominant RAM cost of visual training.

**Data collection (`collect_data.py`)** — `collect_one_step` advances the env by one transition, stores it, and **returns the next observation**, which the caller threads back into the next call (it auto-resets on episode end). `linear_schedule` produces the epsilon value, clipped constant after `max_steps`. Used by the vector-DQN path; `train_flappy.py` inlines its own collection so it can also log episode stats.

**Visual pipeline (`wrappers.py`)** — Flappy-Bird-focused. `make_flappy_env` builds `FlappyBird-rgb-v0` (whose observation is already the RGB screen — no "render to pixels" wrapper needed) and composes, in order: `SkipEnv` (plain action repeat, no max-pooling, only if `frame_skip > 1`) → `GrayscaleWrapper` → `ResizeWrapper` → `FrameStack` → `RecordEpisodeStatistics`. `FrameStack` is always applied and always outputs a contiguous channels-first `(num_stack * C, H, W)` uint8 array — this is what enforces the `CNNQNetwork` input contract. `make_flappy_env` also sets `SDL_VIDEODRIVER`/`SDL_AUDIODRIVER` to `dummy` for headless machines, and passes `disable_env_checker=True` (flappy-bird-gymnasium 0.2.x mis-declares its RGB obs dtype as float32 while returning uint8).

**Visual trainer (`train_flappy.py`)** — `train()` is the tuned pixel-DQN loop: Double DQN target, Huber loss, gradient-norm clipping, reward clipping, a `learning_starts` warmup, one update per `train_freq` env steps, hard target sync every `target_update_interval` env steps, periodic eval + checkpointing. All time-based knobs are counted in **environment steps**. `get_device("auto")` resolves CUDA → Apple MPS → CPU. `record_video()` rolls out a trained agent and writes MP4s of the full-resolution game via OpenCV (`cv2.VideoWriter`, not `moviepy`) from `env.render()` frames, transposing pygame's `(W, H, 3)` surfarray layout to image-style `(H, W, 3)`.

**Evaluation (`evaluation.py`)** — `evaluate_policy` is used by the **vector-DQN** path only (the notebook). It has a Gymnasium-version shim (`VideoRecorder` vs `RecordVideo`) and writes videos to `logs/videos/`. The visual path uses `train_flappy.evaluate` / `record_video` instead.

## Scope constraints

Discrete action spaces only (`spaces.Discrete`). `QNetwork` requires 1D vector observations; `CNNQNetwork` requires channels-first 3D `(C, H, W)` image stacks (use `make_flappy_env`). No continuous control. Double DQN is implemented in the visual trainer; dueling DQN and prioritized replay are not.
