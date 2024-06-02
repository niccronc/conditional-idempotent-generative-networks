import torch
import torch.nn as nn

from CIGN_channel.discriminator_with_channel_conditioning import DiscriminatorWithChannelConditioning
from CIGN_channel.generator_with_channel_conditioning import GeneratorWithChannelConditioning

class CIGNWithChannelConditioning(nn.Module):
    def __init__(self, data_channels=1, latent_dim=64, intermediate_dim=512, embedding_dim=5):
        super(CIGNWithChannelConditioning, self).__init__()
        self.discriminator = DiscriminatorWithChannelConditioning(
            input_channels=data_channels, latent_dim=latent_dim, output_dim=intermediate_dim, embedding_dim=embedding_dim)
        self.generator = GeneratorWithChannelConditioning(
            input_dim=intermediate_dim, output_channels=data_channels, latent_dim=latent_dim, embedding_dim=embedding_dim)

    def forward(self, x, y):
        return self.generator(*self.discriminator(x, y))
