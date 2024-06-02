import torch
import torch.nn as nn

from CIGN_filter.discriminator_with_filter_conditioning import DiscriminatorWithFilterConditioning
from CIGN_filter.generator_with_filter_conditioning import GeneratorWithFilterConditioning

class CIGNWithFilterConditioning(nn.Module):
    def __init__(self, data_channels=1, latent_dim=64, intermediate_dim=512, embedding_dim=5):
        super(CIGNWithFilterConditioning, self).__init__()
        self.discriminator = DiscriminatorWithFilterConditioning(
            input_channels=data_channels, latent_dim=latent_dim, output_dim=intermediate_dim, embedding_dim=embedding_dim)
        self.generator = GeneratorWithFilterConditioning(
            input_dim=intermediate_dim, output_channels=data_channels, latent_dim=latent_dim, embedding_dim=embedding_dim)

    def forward(self, x, y):
        return self.generator(*self.discriminator(x, y))
