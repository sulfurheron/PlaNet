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
        idxes = np.random.choice(len(self.buffer), batch_size, False)
        obss, acts, rews = [], [], []

        for idx in idxes:
            obs, act, rew = self.buffer[idx]
            obss.append(obs)
            acts.append(act)
            rews.append(rew)

        return obss, acts, rews

    def push(
            self, obss: List[ndarray], acts: List[ndarray], rews: List[ndarray]
    ) -> None:
        """Pushes a new observation onto the buffer"""
        obss = torch.tensor(obss)
        acts = torch.tensor(acts)
        rews = torch.tensor(rews).reshape(-1, 1)

        self.buffer.append((obss, acts, rews))
