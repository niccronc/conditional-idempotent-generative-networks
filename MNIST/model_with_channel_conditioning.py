import torch
import torch.nn as nn

from discriminator_with_channel_conditioning import DCGANDiscriminatorWithChannelConditioning
from generator_with_channel_conditioning import DCGANGeneratorWithChannelConditioning

class DCGANLikeModelWithChannelConditioning(nn.Module):
    def __init__(self, data_channels=1, latent_dim=64, intermediate_dim=512, embedding_dim=5):
        super(DCGANLikeModelWithChannelConditioning, self).__init__()
        self.discriminator = DCGANDiscriminatorWithChannelConditioning(
            input_channels=data_channels, latent_dim=latent_dim, output_dim=intermediate_dim, embedding_dim=embedding_dim)
        self.generator = DCGANGeneratorWithChannelConditioning(
            input_dim=intermediate_dim, output_channels=data_channels, latent_dim=latent_dim, embedding_dim=embedding_dim)

    def forward(self, x, y):
        return self.generator(*self.discriminator(x, y))