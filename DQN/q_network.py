import math
from typing import Type

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces


class QNetwork(nn.Module):
    """
    A Q-Network for the DQN algorithm
    to estimate the q-value for a given observation.

    :param observation_space: Observation space of the env,
        contains information about the observation type and shape.
    :param action_space: Action space of the env,
        contains information about the number of actions.
    :param n_hidden_units: Number of units for each hidden layer.
    :param activation_fn: Activation function (ReLU by default)
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        n_hidden_units: int = 64,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        # Assume 1d space
        obs_dim = observation_space.shape[0]
        # Retrieve the number of discrete actions
        n_actions = int(action_space.n)
        # Create the q network (2 fully connected hidden layers)
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim, n_hidden_units),
            activation_fn(),
            nn.Linear(n_hidden_units, n_hidden_units),
            activation_fn(),
            nn.Linear(n_hidden_units, n_actions),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :param observations: A batch of observation (batch_size, obs_dim)
        :return: The Q-values for the given observations
            for all the action (batch_size, n_actions)
        """
        return self.q_net(observations)


class CNNQNetwork(nn.Module):
    """
    Convolutional Q-Network for image observations, following the
    Nature-DQN architecture (Mnih et al., 2015).

    This network is intentionally narrow in scope: it expects **channels-first**
    image stacks of shape ``(C, H, W)``, exactly as produced by
    :func:`DQN.wrappers.make_flappy_env` (e.g. ``(4, 84, 84)`` for a 4-frame
    grayscale stack, or ``(12, 84, 84)`` for a 4-frame RGB stack). There is no
    layout auto-detection: the preprocessing pipeline is the single source of
    truth for the observation format.

    ``uint8`` observation spaces are normalised from ``[0, 255]`` to ``[0, 1]``
    inside :meth:`forward`, so the replay buffer can keep cheap ``uint8`` frames.

    :param observation_space: Observation space of the env. Must be a 3D
        ``(C, H, W)`` ``Box``.
    :param action_space: Action space of the env. Must be ``Discrete``.
    :param n_hidden_units: Number of units in the fully connected layer after
        the convolutional trunk.
    :param activation_fn: Activation function (ReLU by default).
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        n_hidden_units: int = 512,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        obs_shape = observation_space.shape
        assert len(obs_shape) == 3, (
            "CNNQNetwork expects channels-first (C, H, W) observations "
            f"(use make_flappy_env), got shape {obs_shape}"
        )
        n_channels, height, width = obs_shape
        n_actions = int(action_space.n)
        # uint8 frames are normalised to [0, 1] in forward();
        # float observation spaces are assumed to be pre-normalised.
        self._is_uint8 = observation_space.dtype == np.uint8

        # Nature-DQN convolutional trunk.
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            activation_fn(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            activation_fn(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            activation_fn(),
            nn.Flatten(),
        )

        # Infer the flattened conv output size with a dummy forward pass.
        with th.no_grad():
            dummy = th.zeros(1, n_channels, height, width, dtype=th.float32)
            cnn_output_dim = self.cnn(dummy).shape[1]

        # Fully connected head: one hidden layer + the Q-value output layer.
        self.q_head = nn.Sequential(
            nn.Linear(cnn_output_dim, n_hidden_units),
            activation_fn(),
            nn.Linear(n_hidden_units, n_actions),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialisation (as in SB3): it noticeably stabilises
        early training of pixel-based DQN compared to PyTorch defaults."""
        gain = math.sqrt(2.0)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        # Smaller gain on the final Q-value layer keeps initial estimates modest.
        last_linear = self.q_head[-1]
        assert isinstance(last_linear, nn.Linear)
        nn.init.orthogonal_(last_linear.weight, 1.0)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :param observations: A batch of channels-first image stacks,
            shape ``(batch, C, H, W)``.
        :return: Q-values for all actions, shape ``(batch, n_actions)``.
        """
        x = observations.float()
        if self._is_uint8:
            x = x / 255.0
        return self.q_head(self.cnn(x))
