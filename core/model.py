import torch
import torch.nn as nn
import math
import numpy as np

from networks import *
from torch import Tensor
from typing import List, Union


class Model:
    """Abstract world model."""

    def __init__(self) -> None:
        raise NotImplementedError

    def step(self, s: Tensor, a: Tensor, h: Tensor):
        """Step the model one step forward based on the input states."""
        raise NotImplementedError


class DummyModel(Model):
    """Dummy model for testing - Mimics a gym environment."""

    def __init__(self, action_repeat=1) -> None:
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

        self.action_repeat = action_repeat

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

    def step(self, h, s, a):
        """Batched step function"""
        for _ in range(self.action_repeat):
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


class RSSM(nn.Module, Model):
    """Encapsualtes the RSSM model built by the agent."""

    def __init__(
            self, act_dim: int, latent_size, layer_size, hidden_size, device):
        """Initializes all the networks of RSSM."""
        super(RSSM, self).__init__()

        self._LATENT_SIZE = latent_size
        self._LAYER_SIZE = layer_size
        self._HIDDEN_SIZE = hidden_size

        # Initialize all networks
        self._det_state = DeterministicStateModel(
            hidden_size=self._HIDDEN_SIZE, latent_size=self._LATENT_SIZE,
            act_dim=act_dim)
        self._sto_state = StochasticStateModel(
            hidden_size=self._HIDDEN_SIZE, latent_size=self._LATENT_SIZE,
            layer_size=self._LAYER_SIZE)
        self._obs_model = ObservationModel(
            hidden_size=self._HIDDEN_SIZE, latent_size=self._LATENT_SIZE)
        self._rew_model = RewardModel(
            hidden_size=self._HIDDEN_SIZE, latent_size=self._LATENT_SIZE,
            layer_size=self._LAYER_SIZE)
        self._encoder = Encoder(
            hidden_size=self._HIDDEN_SIZE, latent_size=self._LATENT_SIZE,
            layer_size=self._LAYER_SIZE)

        self.device = device
        self.to(device)

    def step(self, h: Tensor, s: Tensor, a: Tensor, get_obs=False):
        """Interfaces with the CEM planner. Batch step the model 1 step forward
        for inference. 

        Args:
            h: Tensor of deterministic states with shape (B, H)
            s: Tensor of stochastic states with shape (B, L)
            a: Tensor of actions with shape (B, A)
            get_obs: Boolean flag of whether or not to return the observation
        """
        h = self._det_state(s.unsqueeze(1), a.unsqueeze(1), h.unsqueeze(0))
        h = h.squeeze()

        s = self._sto_state(h).sample()
        r = self._rew_model(h, s).sample()

        if not get_obs:
            return h, s, r

        o = self._obs_model(h, s).sample()
        return h, s, r, o

    def get_obs_reconstruction_loss(
            self, hs: Tensor, ss: Tensor, obss: Tensor, mask: Tensor) -> Tensor:
        """Computes the differentiable reconstruction loss of observations

        Args:
            hs: Tensor of hidden states with shape (B, T, H) 
            ss: Tensor of states with shape (B, T, L)
            obss: Tensor of observations with shape (B, T, 64, 64, 3)
            mask: Tensor indicating validity with shape (B, T, 1)

            where B is batch size, T is time step, H is hidden size, and
            L is latent size
        """

        # Permute tensors to be time dimension first
        hs = hs.permute(1, 0, 2)
        ss = ss.permute(1, 0, 2)
        obss = obss.permute(1, 0, 2, 3, 4)
        mask = mask.permute(1, 0, 2)

        loss = 0
        T = len(hs)

        for t in range(T):
            h_t, s_t, obs_t, m_t = hs[t], ss[t], obss[t], mask[t]

            # Sample observations
            obs_t_hat = self._obs_model(h_t, s_t).rsample()

            loss_t = -0.5 * ((obs_t_hat - obs_t) ** 2).mean(dim=-1).sum()
            loss_t /= m_t.sum()

            loss += loss_t

        return loss

    def get_rew_reconstruction_loss(
            self, hs: Tensor, ss: Tensor, rews: Tensor, mask: Tensor) -> Tensor:
        """Computes the differentiable reconstruction loss of rewards

        Args:
            hs: Tensor of hidden states with shape (B, T, H) 
            ss: Tensor of states with shape (B, T, L)
            rews: Tensor of rewards with shape (B, T, 1)
            mask: Tensor indicating validity with shape (B, T, 1)

            where B is batch size, T is time step, H is hidden size, and
            L is latent size
        """

        # Permute tensors to be time dimension first
        hs = hs.permute(1, 0, 2)
        ss = ss.permute(1, 0, 2)
        rews = rews.permute(1, 0, 2)
        mask = mask.permute(1, 0, 2)

        loss = 0
        T = len(hs)

        for t in range(T):
            h_t, s_t, r_t, m_t = hs[t], ss[t], rews[t], mask[t]

            # Sample rewards
            r_hat_t = self._rew_model(h_t, s_t).rsample()

            loss_t = -0.5 * ((r_hat_t - r_t) ** 2).sum()
            loss_t /= m_t.sum()

            loss += loss_t

        return loss

    def get_complexity_loss(
            self, hs: Tensor, mus: Tensor, sigmas: Tensor, mask: Tensor
    ) -> Tensor:
        """Computes the differentiable complexity loss

        Args:
            hs: Tensor of hidden states with shape (B, T, H) 
            mus: Tensor of means of the normal distributions computed by the
                encoder with shape (B, T, L)
            sigmas: Tensors of the standard deviations of the normal
                distributions computed by the encoder with shape (B, T, L)
            mask: Tensor indicating validity with shape (B, T, 1)

            where B is batch size, T is time step, H is hidden size, and
            L is latent size
        """

        # Permute tensors to be time dimension first
        hs = hs.permute(1, 0, 2)
        mus = mus.permute(1, 0, 2)
        sigmas = sigmas.permute(1, 0, 2)
        mask = mask.permute(1, 0, 2)

        loss = 0
        T = len(hs)

        assert not torch.isnan(hs).any()
        assert not torch.isnan(mus).any()
        assert not torch.isnan(sigmas).any()
        assert not torch.isnan(mask).any()

        for t in range(T):
            h_t, mu_t, sigma_t, m_t = hs[t], mus[t], sigmas[t], mask[t]
            loss += RSSM._KL_div(mu_t, sigma_t, self._sto_state(h_t), m_t)

        return loss

    def prepare_training(
            self, obss: Tensor, acts: Tensor
    ) -> Union[Tensor, Tensor, Tensor, Tensor]:
        """Returns the up-to-date deterministic and stochastic state estimates
        for the training batch.

        The stochastic states are computed from the previous hidden state and 
        observation using the encoder, and the hidden state is rolled forward
        by the RNN. The initial hidden state is 0.

             a_1     a_2
               \\       \\ 
        h_0      h_1     h_2
           \\   /  \\   /  \\
             s_0     s_1     ...
            /       /       / 
        o_0      o_1     ...

        Args:
            obss: Tensor of observations with shape (B, T, 64, 64, 3)
            acts: Tensor of actions with shape (B, T, A)

        Returns:
            hs: Tensor of hidden states with shape (B, T, H)
            ss: Tensor of stochastic states with shape (B, T, S)
            mus: Tensor of means of the stochastic state distributions with
                shape (B, T, S)
            sigmas: Tensor of stdevs of the stochastic state dist. with
                shape (B, T, S)

            where B is batch size, T is time step, H is hidden size, 
            L is latent size, and A is action size
        """
        obss = obss.permute(1, 0, 2, 3, 4)
        acts = acts.permute(1, 0, 2)

        T, B, _ = acts.shape
        H = self._HIDDEN_SIZE

        hs, ss, mus, sigmas = [], [], [], []
        hs.append(torch.zeros(B, 1, H).to(self.device))

        for t in range(T):
            obs_t, act_t = obss[t], acts[t]  # Shape (B, L) and (B, A) resp.
            h_t = hs[-1].squeeze()  # Current hidden state, shape (B, H)

            dist_t, mu_t, sigma_t = self._encoder(obs_t, h_t)
            s_t = dist_t.rsample()

            # Returns shape (B, 1, H)
            h_tn = self._det_state(
                s_t.unsqueeze(1), act_t.unsqueeze(1), h_t.unsqueeze(0))

            ss.append(s_t.unsqueeze(0))
            mus.append(mu_t.unsqueeze(0))
            sigmas.append(sigma_t.unsqueeze(0))
            hs.append(h_tn)

        hs = hs[:-1]  # Drop the last hidden state

        hs = torch.cat(hs, dim=1)
        ss = torch.cat(ss).permute(1, 0, 2)
        mus = torch.cat(mus).permute(1, 0, 2)
        sigmas = torch.cat(sigmas).permute(1, 0, 2)

        return hs, ss, mus, sigmas

    @staticmethod
    def _KL_div(
            mu_p: Tensor, sigma_p: Tensor, q: Normal, m_t: Tensor
    ) -> Tensor:
        """Computes masked KL(p||q) where both p and q are Gaussians."""

        mu_0, sigma_0 = mu_p * m_t, sigma_p * m_t
        mu_1, sigma_1 = q.loc * m_t, q.scale * m_t

        return 0.5 * (
            (sigma_0 / sigma_1) ** 2 - 1 +
            1 / (sigma_1) * (mu_0 - mu_1) ** 2 +
            2 * torch.log(sigma_1 / sigma_0)).sum() / m_t.sum()


if __name__ == '__main__':
    B = 100
    T = 12
    A = 10

    model = RSSM(A)

    obss = torch.rand(B, T, 64, 64, 3)
    acts = torch.rand(B, T, A)
    rews = torch.rand(B, T, 1)

    hs, ss, mus, sigmas = model.prepare_training(obss, acts)
    model.get_obs_reconstruction_loss(hs, ss, obss)
    model.get_rew_reconstruction_loss(hs, ss, rews)
    model.get_complexity_loss(hs, mus, sigmas)
