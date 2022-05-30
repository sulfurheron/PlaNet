import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.distributions import Normal


class DeterministicStateModel(nn.Module):
    def __init__(self, hidden_size, latent_size, act_dim) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=latent_size + act_dim,
            hidden_size=hidden_size, batch_first=True)

    def forward(self, s: Tensor, a: Tensor):
        x = torch.cat((s, a), dim=1)
        h, _ = self.gru(x)
        return h


class StochasticStateModel(nn.Module):
    def __init__(self, hidden_size, latent_size, layer_size) -> None:
        super().__init__()

        self.layer1 = nn.Linear(hidden_size, layer_size)
        self.layer2 = nn.Linear(layer_size, layer_size)

        self.mu = nn.Linear(layer_size, latent_size)
        self.sigma = nn.Linear(layer_size, latent_size)

    def forward(self, h: Tensor) -> Normal:
        h = F.relu(self.layer1(h))
        h = F.relu(self.layer2(h))

        mu = self.mu(h)
        sigma = F.softplus(self.sigma(h)) + 1e-3

        return Normal(mu, sigma)


class ObservationModel(nn.Module):
    """Deconvolutional network used by the observation model"""

    def __init__(self, hidden_size, latent_size) -> None:
        super().__init__()

        self.deconv1 = nn.ConvTranspose2d(hidden_size + latent_size, 128, 5, 2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, 2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, 2)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 6, 2)

    def forward(self, h: Tensor, s: Tensor) -> Normal:
        x = torch.cat((h, s), dim=1)
        x = x.reshape(*x.shape, 1, 1)

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))

        # N C H W => N H W C
        x = x.permute(0, 2, 3, 1)
        return Normal(x, torch.ones(x.shape))


class RewardModel(nn.Module):
    def __init__(self, hidden_size, latent_size, layer_size) -> None:
        super().__init__()

        self.layer1 = nn.Linear(hidden_size + latent_size, layer_size)
        self.layer2 = nn.Linear(layer_size, layer_size)
        self.output_layer = nn.Linear(layer_size, 1)

    def forward(self, h: Tensor, s: Tensor) -> Normal:
        x = torch.cat((h, s), dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        x = self.output_layer(x)
        return Normal(x, torch.ones(x.shape))


class Dense(nn.Module):
    """Dense feedforward network."""

    def __init__(self, input_dim, output_dim, layer_size) -> None:
        super().__init__()

        self.layer1 = nn.Linear(input_dim, layer_size)
        self.layer2 = nn.Linear(layer_size, layer_size)
        self.output_layer = nn.Linear(layer_size, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output_layer(x)

        return x


class Encoder(nn.Module):
    """Convolutional encoder used by the encoder model."""

    def __init__(self, hidden_size, latent_size, layer_size) -> None:
        super().__init__()

        # Convolutional part of the network
        # Input is 64x64x3 tensor
        self.conv1 = nn.Conv2d(3, 32, 4, 2)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 128, 4, 2)
        self.conv4 = nn.Conv2d(128, 256, 4, 2)
        # Output is 2x2x256 tensor

        # Flatten input for feedforward network
        self.flatten = nn.Flatten()

        # Feedforward part of the network
        self.linear1 = nn.Linear(2 * 2 * 256 + hidden_size, layer_size)
        self.linear2 = nn.Linear(layer_size, layer_size)

        self.mu = nn.Linear(layer_size, latent_size)
        self.sigma = nn.Linear(layer_size, latent_size)

    def forward(self, o: Tensor, h: Tensor) -> Normal:
        """Runs the encoder network.

        Args:
            o: Batched observation with shape (-1, 64, 64, 3)
            h: Batched deterministic model state with shape
                (-1, HIDDEN_STATE_SIZE)
        """

        # N H W C => N C H W
        o = o.permute(0, 3, 1, 2)

        o = F.relu(self.conv1(o))
        o = F.relu(self.conv2(o))
        o = F.relu(self.conv3(o))
        o = F.relu(self.conv4(o))

        o = self.flatten(o)
        x = torch.cat((h, o), dim=1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x)) + 1e-3

        return Normal(mu, sigma)
