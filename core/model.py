import torch
import math
import numpy as np

from torch import Tensor
from typing import Union

device = torch.device('cuda:0')


class Model:
    """Abstract world model."""

    def __init__(self) -> None:
        raise NotImplementedError

    def step(self, s: Tensor, a: Tensor, h: Tensor):
        """Step the model one step forward based on the input states."""
        raise NotImplementedError


class DummyModel(Model):
    """Dummy model for testing - Mimics a gym environment."""

    def __init__(self) -> None:
        self.min_action = torch.tensor([-1.0]).to(device)
        self.max_action = torch.tensor([1.0]).to(device)
        self.max_speed = 0.07
        self.min_position = -1.2
        self.max_position = 0.6
        self.goal_position = (
            0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        )
        self.goal_velocity = 0
        self.power = torch.tensor(0.0015)

        self.low_state = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )

        self.max_speed = torch.tensor([0.07]).to(device)
        self.min_position = torch.tensor([-1.2]).to(device)
        self.max_position = torch.tensor([0.6]).to(device)

    def _step(self, s: Tensor, a: Tensor, h):
        position, velocity = s[0], s[1]
        force = min(max(a[0], self.min_action), self.max_action)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if velocity > self.max_speed:
            velocity = self.max_speed
        if velocity < -self.max_speed:
            velocity = -self.max_speed
        position += velocity
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = 0

    def step(self, s, a, h):
        """Batched step function"""
        position, velocity = s[:, [0]], s[:, [1]]
        force = torch.minimum(
            torch.maximum(a, self.min_action), self.max_action)

        velocity += force * self.power - 0.0025 * torch.cos(3 * position)
        velocity = torch.minimum(velocity, self.max_speed)
        velocity = torch.maximum(velocity, -self.max_speed)

        position += velocity
        position = torch.minimum(position, self.max_position)
        position = torch.maximum(position, self.min_position)

        velocity = torch.where(
            torch.logical_and(position == self.min_position, velocity < 0),
            torch.zeros(velocity.shape).to(device), velocity)

        # Convert a possible numpy bool to a Python bool.
        done = torch.logical_and(
            position >= self.goal_position, velocity >= self.goal_velocity)

        reward = -(a ** 2) * 0.1 + done * 100
        position = position.reshape(-1, 1)
        velocity = velocity.reshape(-1, 1)
        s = torch.cat((position, velocity), dim=1)
        return h, s, reward


class RSSM(Model):
    """Encapsualtes the RSSM model built by the agent."""

    def __init__(self) -> None:
        """Initializes all the networks of RSSM."""
        pass

    def step(self, s: Tensor, a: Tensor, h: Tensor) \
            -> Union[Tensor, Tensor, Tensor]:
        pass
