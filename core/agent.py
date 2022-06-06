import random
import torch
import gym
import numpy as np

from torch import Tensor
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

from model import RSSM
from buffer import Buffer
from planner import CEMPlanner

from tqdm import trange


class Agent:
    """An RL Agent"""

    def __init__(
            self, env: gym.Env, seed_ep=5, batch_size=50, chunk_len=50, lr=1e-3,
            planning_horizon=12, optimization_iter=10, candidates_per_iter=1000,
            num_fit_candidates=100, latent_size=30, layer_size=200,
            hidden_size=200, device=torch.device('cuda:0')
    ) -> None:
        self.env = env
        self.batch_size = batch_size
        self.chunk_len = chunk_len

        self.latent_size = latent_size
        self.layer_size = layer_size
        self.hidden_size = hidden_size

        self.candidates_per_iter = candidates_per_iter

        self.act_dim = env.action_space.shape[0]

        self.model = RSSM(
            self.act_dim, latent_size, layer_size, hidden_size, device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr, eps=1e-4)
        self.planner = CEMPlanner(
            self.model, self.act_dim, planning_horizon, optimization_iter,
            candidates_per_iter, num_fit_candidates, device)

        # Initialize random seed episodes
        self.buffer = Buffer()
        for _ in trange(seed_ep, desc='Sampling Trajectories'):
            obss, acts, rews = [], [], []

            obs = self.env.reset()
            done = False

            while not done:
                act = self.env.action_space.sample()
                nobs, rew, done, _ = self.env.step(act)

                obss.append(torch.tensor(obs))
                acts.append(torch.tensor(act))
                rews.append(torch.tensor(rew))

                obs = nobs

            self.buffer.push(obss, acts, rews)

        self.device = device

        self.writer = SummaryWriter(flush_secs=30)
        self.tick = 0

        # JIT trace the loss functions
        B = self.batch_size
        T = self.chunk_len
        L = self.latent_size
        H = self.hidden_size

        self.traced_model = torch.jit.trace_module(
            mod=self.model,
            inputs={
                'get_obs_reconstruction_loss': (
                    torch.zeros(B, T, H, device=self.device),
                    torch.zeros(B, T, L, device=self.device),
                    torch.zeros(B, T, 64, 64, 3, device=self.device),
                    torch.zeros(B, T, 1, device=self.device)
                ),
                'get_rew_reconstruction_loss': (
                    torch.zeros(B, T, H, device=self.device),
                    torch.zeros(B, T, L, device=self.device),
                    torch.zeros(B, T, 1, device=self.device),
                    torch.zeros(B, T, 1, device=self.device)
                ),
                'get_complexity_loss': (
                    torch.zeros(B, T, H, device=self.device),
                    torch.zeros(B, T, L, device=self.device),
                    torch.zeros(B, T, L, device=self.device),
                    torch.zeros(B, T, 1, device=self.device)
                ),
            }
        )

    def fit_model(self):
        """Performs model fitting. Draws sequence chunks and updates."""
        obss, _, rews, hs, ss, mus, sigmas, mask = \
            self._sample_and_prepare_chunks()

        loss = self.traced_model.get_obs_reconstruction_loss(hs, ss, obss, mask)
        loss += self.traced_model.get_rew_reconstruction_loss(
            hs, ss, rews, mask)
        loss -= 2 * self.traced_model.get_complexity_loss(hs, mus, sigmas, mask)

        self.writer.add_scalar('Loss/train', -float(loss), self.tick)
        self.tick += 1

        self.opt.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1000.0)
        loss *= -1  # Turn gradient descent into gradient ascent
        loss.backward()
        self.opt.step()

    def _sample_and_prepare_chunks(self):
        """Samples sequences from buffers.

        Computes all the hidden states and take random chunks from the sampled
        sequences. Pad sequences with 0s when necessary and returns a mask
        indicating the validity.
        """

        _obss, _acts, _rews = self.buffer.sample_sequence(self.batch_size)
        seq_lengths = torch.tensor([len(obs) for obs in _obss]).reshape(-1, 1)

        _obss = pad_sequence(_obss, batch_first=True).to(self.device)
        _acts = pad_sequence(_acts, batch_first=True).to(self.device)
        _rews = pad_sequence(_rews, batch_first=True).to(self.device)

        # Roll the model forward for each (padded) sequence. Required since
        # the states are history-dependent, so we have to roll from the start.
        _hs, _ss, _mus, _sigmas = self.model.prepare_training(_obss, _acts)

        # Move tensors out of GPU to save memory
        cpu = torch.device('cpu')
        _obss = _obss.to(cpu)
        _acts = _acts.to(cpu)
        _rews = _rews.to(cpu)

        padded_len = _obss.shape[1]
        start = random.randint(0, padded_len - self.chunk_len)
        end = start + self.chunk_len

        def select(tensor):
            return tensor[:, start:end].to(self.device)

        obss = select(_obss)
        acts = select(_acts)
        rews = select(_rews)
        hs = select(_hs)
        ss = select(_ss)
        mus = select(_mus)
        sigmas = select(_sigmas)

        mask = torch.arange(start, end).repeat(self.batch_size, 1)
        mask = (mask < seq_lengths).to(torch.float32).to(self.device)
        mask = mask.reshape(*mask.shape, 1)

        return obss, acts, rews, hs, ss, mus, sigmas, mask

    def evaluate(self, num_repeats=5):
        """Evaluate the current agent"""
        rewards = []
        for _ in trange(5, desc='Evaluating', leave=False):
            *_, rews = self._inference(lambda x: x)
            rewards.append(float(torch.tensor(rews).sum()))

        self.writer.add_scalar('Rewards/test', np.mean(rewards), self.tick)

    def collect_data(self):
        """Collects data using the current model"""

        # Exploration noise
        exp_noise = Normal(
            torch.zeros(self.act_dim, device=self.device),
            0.3 * torch.ones(self.act_dim, device=self.device))

        obss, acts, rews = self._inference(lambda x: x + exp_noise.sample())
        self.buffer.push(obss, acts, rews)

        self.writer.add_scalar(
            'Rewards/train', float(torch.tensor(rews).sum()), self.tick)

    def _inference(self, callback):
        # CPU device
        cpu = torch.device('cpu')

        obss, acts, rews = [], [], []
        obss.append(torch.tensor(self.env.reset()))
        h = torch.zeros(1, self.hidden_size).to(self.device)

        done = False
        while not done:
            obs = obss[-1].unsqueeze(0).to(self.device)  # (B=1, 64, 64, 3)
            dist, *_ = self.model._encoder(obs, h)

            mu = self.planner.plan(dist, h)
            mu = callback(mu)
            acts.append(mu.to(cpu))  # Do not store GPU arrays in buffer

            n_obs, rew, done, _ = self.env.step(mu.detach().cpu().numpy())
            obss.append(torch.tensor(n_obs).to(cpu))
            rews.append(torch.tensor(rew).to(cpu))

            s = dist.sample().unsqueeze(1)  # (B, T, L)
            a = mu.reshape(1, 1, -1)  # (B, T, A)

            h = self.model._det_state(s, a, h.unsqueeze(0)).squeeze(1)

        return obss[:-1], acts, rews
