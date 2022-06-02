import torch
import numpy as np

from torch import Tensor
from typing import List, Union
from numpy import ndarray


class Buffer:
    """Buffer holding prior experiences."""

    def __init__(self) -> None:
        """Initializes an empty buffer."""
        self.buffer = []

    def sample_sequence(self, batch_size: int) -> Union[
        List[Tensor], List[Tensor], List[Tensor]]:
        """Samples a consecutive sequence of observations."""
        idxes = np.random.choice(len(self.buffer), batch_size, True)

        obss, acts, rews = [], [], []

        for idx in idxes:
            obs, act, rew = self.buffer[idx]
            obss.append(obs)
            acts.append(act)
            rews.append(rew)

        return obss, acts, rews

    def push(
            self, obss: List[Tensor], acts: List[Tensor], rews: List[Tensor]
    ) -> None:
        """Pushes a new observation onto the buffer"""
        obss = torch.stack(obss)
        acts = torch.stack(acts)
        rews = torch.stack(rews).reshape(-1, 1)

        assert len(obss.shape) == 4
        assert len(acts.shape) == 2
        assert len(rews.shape) == 2

        self.buffer.append((obss, acts, rews))
