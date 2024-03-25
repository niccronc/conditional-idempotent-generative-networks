import torch
import torch.nn as nn

class DCGANDiscriminatorWithChannelConditioning(nn.Module):
    def __init__(self, input_channels=1, latent_dim=64, output_dim=512, embedding_dim=5):
        super(DCGANDiscriminatorWithChannelConditioning, self).__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim

        # image layers

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.final_activation = nn.Sigmoid()

        self.conv_1 = nn.Conv2d(self.input_channels + 1, self.latent_dim, 4, 2, 1, bias=False)
        self.conv_2 = nn.Conv2d(self.latent_dim + 1, self.latent_dim * 2, 4, 2, 1, bias=False)
        self.conv_3 = nn.Conv2d(self.latent_dim * 2 + 1, self.latent_dim * 4, 3, 2, 1, bias=False)
        self.conv_4 = nn.Conv2d(self.latent_dim * 4 + 1, self.latent_dim * 8, 4, 2, 1, bias=False)
        self.conv_5 = nn.Conv2d(self.latent_dim * 8 + 1, self.output_dim, 2, 1, 0, bias=False)

        self.dropout_1 = nn.Dropout(0.15)
        self.dropout_2 = nn.Dropout(0.15)
        self.dropout_3 = nn.Dropout(0.15)
        self.dropout_4 = nn.Dropout(0.15)
        self.dropout_5 = nn.Dropout(0.15)

        self.batch_norm_1 = nn.BatchNorm2d(self.latent_dim)
        self.batch_norm_2 = nn.BatchNorm2d(self.latent_dim * 2)
        self.batch_norm_3 = nn.BatchNorm2d(self.latent_dim * 4)
        self.batch_norm_4 = nn.BatchNorm2d(self.latent_dim * 8)

        # labels layers

        self.label_embedder = nn.Embedding(num_embeddings = 10, embedding_dim = self.embedding_dim)
        self.linear_1 = nn.Linear(self.embedding_dim, 28 * 28)
        self.linear_2 = nn.Linear(self.embedding_dim, 14 * 14)
        self.linear_3 = nn.Linear(self.embedding_dim, 7 * 7)
        self.linear_4 = nn.Linear(self.embedding_dim, 4 * 4)
        self.linear_5 = nn.Linear(self.embedding_dim, 2 * 2)

        self.label_activation_function = nn.Tanh()


    def forward(self, input, label):

        label_embedded = self.label_embedder(label)
        label_contribution_layer_1 = self.label_activation_function(self.linear_1(label_embedded).view(-1, 28, 28))
        label_contribution_layer_2 = self.label_activation_function(self.linear_2(label_embedded).view(-1, 14, 14))
        label_contribution_layer_3 = self.label_activation_function(self.linear_3(label_embedded).view(-1, 7, 7))
        label_contribution_layer_4 = self.label_activation_function(self.linear_4(label_embedded).view(-1, 4, 4))
        label_contribution_layer_5 = self.label_activation_function(self.linear_5(label_embedded).view(-1, 2, 2))

        # input is 1 x 28 x 28 for MNIST
        output = self.conv_1(torch.cat((input, label_contribution_layer_1.unsqueeze(1)), dim=1)) #self.conv_1(input)
        output = self.dropout_1(output)
        output = self.activation(output)

        # shape is latent_dim x 14 x 14
        output = self.batch_norm_1(output)
        output = self.conv_2(torch.cat((output, label_contribution_layer_2.unsqueeze(1)), dim=1)) #self.conv_2(output)
        output = self.dropout_2(output)
        output = self.activation(output)

        # shape is (latent_dim*2) x 7 * 7
        output = self.batch_norm_2(output)
        output = self.conv_3(torch.cat((output, label_contribution_layer_3.unsqueeze(1)), dim=1)) #self.conv_3(output)
        output = self.dropout_3(output)
        output = self.activation(output)

        # shape is (latent_dim*4) x 4 x 4
        output = self.batch_norm_3(output)
        output = self.conv_4(torch.cat((output, label_contribution_layer_4.unsqueeze(1)), dim=1)) #self.conv_4(output)
        output = self.dropout_4(output)
        output = self.activation(output)

        # shape is (latent_dim*8) x 2 x 2
        output = self.batch_norm_4(output)
        output = self.conv_5(torch.cat((output, label_contribution_layer_5.unsqueeze(1)), dim=1)) #self.conv_5(output)
        output = self.dropout_5(output)
        output = self.final_activation(output)

        # shape is output_dim x 1 x 1

        return output, label