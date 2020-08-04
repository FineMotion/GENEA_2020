import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
HIDDEN_DIM = 40
INPUT_DIM = 45
OUTPUT_DIM = 20


class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder_hidden = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.encoder_mu = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        self.encoder_sigma = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        self.decoder_hidden = nn.Linear(OUTPUT_DIM, HIDDEN_DIM)
        self.decoder_out = nn.Linear(HIDDEN_DIM, INPUT_DIM)

    def encode(self, x):
        x = self.encoder_hidden(x)
        x = torch.relu(x)
        return self.encoder_mu(x), self.encoder_sigma(x)

    def reparametrize(self, mu, sigma):
        std = torch.exp(0.5*sigma)
        eps = torch.rand_like(std)
        return mu + eps*std

    def decode(self, z):
        z = self.decoder_hidden(z)
        z = torch.relu(z)
        z = self.decoder_out(z)
        return torch.tanh(z)

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparametrize(mu, sigma)
        return self.decode(z), mu, sigma


class VAELoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(VAELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, outputs, labels):
        reconstruct, mu, sigma = outputs
        recons_loss = F.mse_loss(reconstruct, labels)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + sigma - mu ** 2 - sigma.exp(), dim=1), dim=0)
        return recons_loss + kld_loss
