from typing import Union

import gym
import torch

from torch import Tensor
from torch.distributions import Distribution, Normal
from model import Model, DummyModel
from wrappers import ActionRepeatWrapper

from tqdm import trange


class CEMPlanner:
    """Cross Entropy Planner"""

    def __init__(
            self,
            model: Model,
            act_dim: int,
            planning_horizon: int,
            optimization_iter: int,
            candidates_per_iter: int,
            num_fit_candidates: int,
            device,
    ) -> None:

        self._model = model
        self._act_dim = act_dim
        self._planning_horizon = planning_horizon
        self._optimization_iter = optimization_iter
        self._candidates_per_iter = candidates_per_iter
        self._num_fit_candidates = num_fit_candidates

        self.device = device

    @torch.no_grad()
    def plan(self, state: Distribution, h: Tensor) -> Tensor:
        """Plans future actions from a given stochastic state (distribution) and
        a deterministic state.

        Args:
            state: Distribution of stochastic states where each sample produces
                a tensor of shape (1, latent_size)
            h: Tensor of deterministic state with shape (1, hidden_size)
        """
        mu = torch.zeros(
            self._planning_horizon, self._act_dim).to(self.device)
        sigma = torch.ones(
            self._planning_horizon, self._act_dim).to(self.device)
        act_dist = Normal(mu, sigma)

        # Massage h into shape (B, H)
        h = h.repeat(self._candidates_per_iter, 1)

        # for _ in trange(
        #         self._optimization_iter, desc='Optimization',
        #         position=2, leave=False):
        for _ in range(self._optimization_iter):
            rewards = torch.zeros(self._candidates_per_iter, 1).to(self.device)

            act_seqs = act_dist.sample(
                (self._candidates_per_iter,)).to(self.device)
            s = state.sample(
                (self._candidates_per_iter,)).to(self.device).squeeze()

            for t in range(self._planning_horizon):
                h, s, r = self._model.step(h, s, act_seqs[:, t])
                rewards += r

            # Refit belief
            idxes = torch.argsort(
                rewards.reshape(-1), descending=True).to(self.device)
            idxes = idxes[:self._num_fit_candidates]

            mu = act_seqs[idxes].mean(dim=0)
            sigma = (act_seqs[idxes] - mu).abs().sum(dim=0) / (len(idxes) - 1)

            act_dist = Normal(mu, sigma)

        return mu[0]


if __name__ == '__main__':
    model = DummyModel(action_repeat=2)
    planner = CEMPlanner(model, 1, 500, 10, 1000, 100)

    class DummyState:
        def __init__(self, s=None) -> None:
            self.state = s if s is not None else [0, 0]

        def sample(self, n):
            state = torch.tensor(
                self.state, dtype=torch.float32).reshape(1, -1)
            return state.repeat(n[0], 1)

    def get_action(state):
        state = DummyState(state)
        return planner.plan(state, None)

    env = ActionRepeatWrapper(gym.make('MountainCarContinuous-v0'), 2)

    state = env.reset()
    done = False

    total_reward = 0
    ep_len = 0

    while not done:
        env.render()
        action = get_action(state).cpu().numpy()
        state, reward, done, info = env.step(action)

        total_reward += reward
        ep_len += 1

        print(ep_len, action, total_reward)
