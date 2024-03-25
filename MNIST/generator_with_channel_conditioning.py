import torch
import torch.nn as nn

class DCGANGeneratorWithChannelConditioning(nn.Module):
    def __init__(self, input_dim=512, output_channels=1, latent_dim=64, embedding_dim=5):
        super(DCGANGeneratorWithChannelConditioning, self).__init__()
        self.input_dim = input_dim
        self.output_channels = output_channels
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim

        # image layers

        self.activation = nn.ReLU(True)
        self.final_activation = nn.Tanh()

        self.conv_1 = nn.ConvTranspose2d(self.input_dim + 1, self.latent_dim * 8, 2, 1, 0, bias=False)
        self.conv_2 = nn.ConvTranspose2d(self.latent_dim * 8 + 1, self.latent_dim * 4, 4, 2, 1, bias=False)
        self.conv_3 = nn.ConvTranspose2d(self.latent_dim * 4 + 1, self.latent_dim * 2, 3, 2, 1, bias=False)
        self.conv_4 = nn.ConvTranspose2d(self.latent_dim * 2 + 1, self.latent_dim, 4, 2, 1, bias=False)
        self.conv_5 = nn.ConvTranspose2d(self.latent_dim + 1, self.output_channels, 4, 2, 1, bias=False)

        self.dropout_1 = nn.Dropout(0.15)
        self.dropout_2 = nn.Dropout(0.15)
        self.dropout_3 = nn.Dropout(0.15)
        self.dropout_4 = nn.Dropout(0.15)

        self.batch_norm_1 = nn.BatchNorm2d(self.input_dim)
        self.batch_norm_2 = nn.BatchNorm2d(self.latent_dim * 8)
        self.batch_norm_3 = nn.BatchNorm2d(self.latent_dim * 4)
        self.batch_norm_4 = nn.BatchNorm2d(self.latent_dim * 2)
        self.batch_norm_5 = nn.BatchNorm2d(self.latent_dim)

        # labels layers

        self.label_embedder = nn.Embedding(num_embeddings = 10, embedding_dim = self.embedding_dim)
        self.linear_1 = nn.Linear(self.embedding_dim, 1 * 1)
        self.linear_2 = nn.Linear(self.embedding_dim, 2 * 2)
        self.linear_3 = nn.Linear(self.embedding_dim, 4 * 4)
        self.linear_4 = nn.Linear(self.embedding_dim, 7 * 7)
        self.linear_5 = nn.Linear(self.embedding_dim, 14 * 14)

        self.label_activation_function = nn.Tanh()

    def forward(self, input, label):

        label_embedded = self.label_embedder(label)
        label_contribution_layer_1 = self.label_activation_function(self.linear_1(label_embedded).view(-1, 1, 1))
        label_contribution_layer_2 = self.label_activation_function(self.linear_2(label_embedded).view(-1, 2, 2))
        label_contribution_layer_3 = self.label_activation_function(self.linear_3(label_embedded).view(-1, 4, 4))
        label_contribution_layer_4 = self.label_activation_function(self.linear_4(label_embedded).view(-1, 7, 7))
        label_contribution_layer_5 = self.label_activation_function(self.linear_5(label_embedded).view(-1, 14, 14))

        # input is input_dim x 1 x 1
        output = self.batch_norm_1(input)
        output = self.conv_1(torch.cat((output, label_contribution_layer_1.unsqueeze(1)), dim=1)) #self.conv_1(output)
        output = self.dropout_1(output)
        output = self.activation(output)

        # shape is (latent_dim*8) x 2 x 2
        output = self.batch_norm_2(output)
        output = self.conv_2(torch.cat((output, label_contribution_layer_2.unsqueeze(1)), dim=1)) #self.conv_2(output)
        output = self.dropout_2(output)
        output = self.activation(output)

        # shape is (latent_dim*4) x 4 * 4
        output = self.batch_norm_3(output)
        output = self.conv_3(torch.cat((output, label_contribution_layer_3.unsqueeze(1)), dim=1)) #self.conv_3(output)
        output = self.dropout_3(output)
        output = self.activation(output)

        # shape is (latent_dim*2) x 7 x 7
        output = self.batch_norm_4(output)
        output = self.conv_4(torch.cat((output, label_contribution_layer_4.unsqueeze(1)), dim=1)) #self.conv_4(output)
        output = self.dropout_4(output)
        output = self.activation(output)

        # shape is (latent_dim) x 14 x 14
        output = self.batch_norm_5(output)
        output = self.conv_5(torch.cat((output, label_contribution_layer_5.unsqueeze(1)), dim=1)) #self.conv_5(output)
        output = self.final_activation(output)

        # shape is output_channels x 28 x 28

        return output, label