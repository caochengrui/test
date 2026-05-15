"""
Environment wrappers and factory for visual (pixel-based) DQN on Flappy Bird.

The whole module has a single job: turn the raw RGB game screen of
``FlappyBird-rgb-v0`` into a compact, **channels-first**, CNN-ready stack of
preprocessed frames. The pipeline is::

    raw RGB (H, W, 3) uint8
        -> SkipEnv          optional action repeat (no max-pooling)
        -> GrayscaleWrapper (H, W) uint8
        -> ResizeWrapper    (84, 84) uint8
        -> FrameStack       (n_stack, 84, 84) uint8   <- channels-first, CNN-ready
        -> RecordEpisodeStatistics

The output of :func:`make_flappy_env` plugs directly into
:class:`DQN.q_network.CNNQNetwork` and :class:`DQN.replay_buffer.ReplayBuffer`.

This is deliberately Flappy-Bird-focused. CartPole-style vector environments
should use the MLP path (:class:`DQN.q_network.QNetwork`) instead — rendering
CartPole to pixels throws away the very state that makes it learnable.
"""

import os
from collections import deque
from typing import Tuple

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SkipEnv(gym.Wrapper):
    """
    Repeat each action for ``skip`` env steps, summing the rewards and
    returning the **last** observed frame.

    Unlike Atari's ``MaxAndSkipEnv`` there is no max-pooling over consecutive
    frames: the Flappy Bird renderer does not flicker, so the last frame is
    already a clean observation. ``skip=1`` makes this wrapper a no-op.

    :param env: The environment to wrap.
    :param skip: Number of env steps each action is repeated over.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        assert skip >= 1, f"skip must be >= 1, got {skip}"
        self._skip = skip

    def step(self, action):
        """Repeat ``action`` for ``skip`` steps; return summed reward + last frame."""
        total_reward = 0.0
        terminated = truncated = False
        obs = None
        info: dict = {}
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class GrayscaleWrapper(gym.ObservationWrapper):
    """
    Convert an ``(H, W, 3)`` RGB observation to an ``(H, W)`` grayscale one.

    :param env: The environment to wrap. Must emit ``(H, W, 3)`` observations.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        old_space = env.observation_space
        assert isinstance(old_space, spaces.Box), "Observation space must be Box"
        assert len(old_space.shape) == 3 and old_space.shape[2] == 3, (
            f"GrayscaleWrapper expects (H, W, 3) observations, got shape {old_space.shape}"
        )
        h, w = old_space.shape[0], old_space.shape[1]
        self.observation_space = spaces.Box(low=0, high=255, shape=(h, w), dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Convert an RGB frame to a single-channel grayscale frame."""
        return cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)


class ResizeWrapper(gym.ObservationWrapper):
    """
    Resize image observations to a target ``(height, width)`` with area
    interpolation. Works on both ``(H, W)`` and ``(H, W, C)`` observations.

    :param env: The environment to wrap.
    :param shape: Target size as ``(height, width)``.
    """

    def __init__(self, env: gym.Env, shape: Tuple[int, int] = (84, 84)) -> None:
        super().__init__(env)
        assert len(shape) == 2, f"shape must be (height, width), got {shape}"
        self.target_h, self.target_w = shape

        old_space = env.observation_space
        assert isinstance(old_space, spaces.Box), "Observation space must be Box"

        if len(old_space.shape) == 2:
            new_shape: Tuple[int, ...] = (self.target_h, self.target_w)
        elif len(old_space.shape) == 3:
            new_shape = (self.target_h, self.target_w, old_space.shape[2])
        else:
            raise ValueError(f"Unexpected observation shape: {old_space.shape}")

        self.observation_space = spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Resize the observation to ``(target_h, target_w)``."""
        return cv2.resize(
            observation,
            (self.target_w, self.target_h),
            interpolation=cv2.INTER_AREA,
        )


class FrameStack(gym.Wrapper):
    """
    Stack the last ``num_stack`` frames into a single **channels-first** array.

    The output is always a plain ``np.uint8`` array of shape
    ``(num_stack * C, H, W)`` — with ``C = 1`` for grayscale input and ``C = 3``
    for RGB input — so it is directly consumable by a CNN and by the replay
    buffer. ``num_stack=1`` is allowed and simply adds the channel axis
    (``(H, W)`` -> ``(1, H, W)``).

    Unlike ``gymnasium.wrappers.FrameStackObservation`` (which may hand back
    ``LazyFrames``), this always returns a contiguous numpy array.

    :param env: The environment to wrap.
    :param num_stack: Number of frames to stack.
    """

    def __init__(self, env: gym.Env, num_stack: int = 4) -> None:
        super().__init__(env)
        assert num_stack >= 1, f"num_stack must be >= 1, got {num_stack}"
        self.num_stack = num_stack
        self.frames: deque = deque(maxlen=num_stack)

        old_space = env.observation_space
        assert isinstance(old_space, spaces.Box), "Observation space must be Box"
        if len(old_space.shape) == 2:
            self._channels = 1
            h, w = old_space.shape
        elif len(old_space.shape) == 3:
            h, w, self._channels = old_space.shape
        else:
            raise ValueError(
                f"FrameStack expects 2D or 3D observations, got shape {old_space.shape}"
            )

        n_channels = num_stack * self._channels
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(n_channels, h, w), dtype=np.uint8
        )

    @staticmethod
    def _to_chw(frame: np.ndarray) -> np.ndarray:
        """Add/move the channel axis to the front: (H,W)->(1,H,W), (H,W,C)->(C,H,W)."""
        if frame.ndim == 2:
            return frame[np.newaxis, ...]
        return np.transpose(frame, (2, 0, 1))

    def _get_obs(self) -> np.ndarray:
        """Concatenate the buffered frames along the channel axis."""
        return np.concatenate(list(self.frames), axis=0).astype(np.uint8)

    def reset(self, **kwargs):
        """Reset the env and fill the stack with copies of the first frame."""
        obs, info = self.env.reset(**kwargs)
        chw = self._to_chw(obs)
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(chw)
        return self._get_obs(), info

    def step(self, action):
        """Step the env and push the new frame onto the stack."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(self._to_chw(obs))
        return self._get_obs(), reward, terminated, truncated, info


def _ensure_headless(render_mode) -> None:
    """
    Make pygame (used by flappy-bird-gymnasium) usable on a headless machine
    such as Colab or a GPU server with no display.

    Uses ``setdefault`` so an explicitly configured environment is left alone,
    and skips the video stub when the caller actually wants a human window.
    """
    if render_mode != "human":
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # A dummy audio driver turns sound loading/playback into no-ops, which
    # avoids ALSA errors on servers without a sound device.
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def make_flappy_env(
    env_id: str = "FlappyBird-rgb-v0",
    frame_skip: int = 1,
    grayscale: bool = True,
    resize_shape: Tuple[int, int] = (84, 84),
    frame_stack: int = 4,
    render_mode=None,
    record_episode_stats: bool = True,
    seed=None,
    **env_kwargs,
) -> gym.Env:
    """
    Build a Flappy Bird environment with pixel observations and the standard
    DQN preprocessing pipeline.

    The base environment is ``FlappyBird-rgb-v0`` from ``flappy-bird-gymnasium``,
    whose observation is already the rendered RGB game screen — so no
    "render to pixels" wrapper is needed. The pipeline applied (in order):

    1. ``SkipEnv``            — action repeat (only if ``frame_skip > 1``).
    2. ``GrayscaleWrapper``   — RGB -> grayscale (only if ``grayscale``).
    3. ``ResizeWrapper``      — resize to ``resize_shape`` (only if not ``None``).
    4. ``FrameStack``         — stack ``frame_stack`` frames, channels-first.
    5. ``RecordEpisodeStatistics`` — track episode return/length in ``info``.

    Final observation shape: ``(frame_stack, *resize_shape)`` for grayscale,
    or ``(frame_stack * 3, *resize_shape)`` for RGB.

    :param env_id: Gymnasium environment id (defaults to ``FlappyBird-rgb-v0``).
    :param frame_skip: Number of env steps each action is repeated (1 = no skip).
    :param grayscale: Whether to convert frames to grayscale.
    :param resize_shape: Target ``(height, width)``; ``None`` disables resizing.
    :param frame_stack: Number of frames to stack (>= 1).
    :param render_mode: Base-env render mode (use ``"rgb_array"`` for video).
    :param record_episode_stats: Wrap with ``RecordEpisodeStatistics``.
    :param seed: If given, seeds the action space (useful for reproducible
        epsilon-greedy exploration). The observation stream should still be
        seeded by the caller via ``env.reset(seed=...)``.
    :param env_kwargs: Extra keyword arguments forwarded to ``gym.make``
        (e.g. ``audio_on=False`` if a future env build requires it).
    :return: The fully wrapped environment.
    """
    _ensure_headless(render_mode)

    try:
        import flappy_bird_gymnasium  # noqa: F401  (registers FlappyBird-* envs)
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "flappy-bird-gymnasium is required for the visual DQN. Install it with:\n"
            '  pip install "flappy-bird-gymnasium @ '
            'git+https://github.com/araffin/flappy-bird-gymnasium@patch-1"'
        ) from exc

    # ``audio_on=False``: the flappy-bird-gymnasium renderer calls
    # ``pygame.mixer.init()`` at construction, which fails on a headless server
    # with no audio device (a dummy SDL audio driver is not always enough).
    # Disabling audio skips sound loading entirely; the caller can still
    # override it via ``env_kwargs``.
    env_kwargs.setdefault("audio_on", False)
    # ``disable_env_checker=True``: flappy-bird-gymnasium 0.2.x declares the RGB
    # observation space with a float32 dtype but actually returns uint8 frames,
    # which makes the passive env checker spam warnings. The preprocessing
    # wrappers below redefine the observation space as uint8 anyway.
    env = gym.make(env_id, render_mode=render_mode, disable_env_checker=True, **env_kwargs)

    # 1. Action repeat.
    if frame_skip > 1:
        env = SkipEnv(env, skip=frame_skip)

    # 2. Grayscale.
    if grayscale:
        env = GrayscaleWrapper(env)

    # 3. Resize.
    if resize_shape is not None:
        env = ResizeWrapper(env, shape=resize_shape)

    # 4. Frame stacking (always applied: it also enforces the channels-first
    #    layout the CNN expects, even when frame_stack == 1).
    env = FrameStack(env, num_stack=frame_stack)

    # 5. Episode statistics.
    if record_episode_stats:
        env = gym.wrappers.RecordEpisodeStatistics(env)

    if seed is not None:
        env.action_space.seed(seed)

    return env
