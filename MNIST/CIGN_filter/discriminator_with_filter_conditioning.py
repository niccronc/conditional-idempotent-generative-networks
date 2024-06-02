import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
        
def batch_channel_wise_2d_convolution(image, kernel, stride, padding, bias=None):
    import math
    '''
    Implements batch-wise and channel-wise cross correlation between:
    an image of size (B, C, H, W) and
    a kernel of size (B, C, h, w). In particular, batch and channels are expected to match
    '''
    B, C_in, H, W = image.shape
    b, c_in, h, w = kernel.shape    

    H_out = math.floor(((H + 2 * padding - (h - 1) - 1) / stride) + 1)
    W_out = math.floor(((W + 2 * padding - (w - 1) - 1) / stride) + 1)
    
    result = F.conv2d(
        image.contiguous().view(1, B * C_in, H, W),
        kernel.contiguous().view(b*c_in, 1, h, w),
        stride = stride,
        padding = padding,
        bias = bias,
        groups = B * C_in
        ).reshape(b, c_in, H_out, W_out)
    
    #torch.cuda.empty_cache()
    #gc.collect()

    return result

class DiscriminatorWithFilterConditioning(nn.Module):

    def __init__(self, 
                 input_channels=1, 
                 latent_dim=64, 
                 output_dim=512, 
                 embedding_dim=5,
                 dropout_prob = 0.15
                ):
        super(DiscriminatorWithFilterConditioning, self).__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout_prob

        # image layers
        
        self.kernel_1 = 4
        self.kernel_2 = 4
        self.kernel_3 = 3
        self.kernel_4 = 4
        self.kernel_5 = 2

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.final_activation = nn.Sigmoid()

        self.dropout_1 = nn.Dropout(self.dropout_prob)
        self.dropout_2 = nn.Dropout(self.dropout_prob)
        self.dropout_3 = nn.Dropout(self.dropout_prob)
        self.dropout_4 = nn.Dropout(self.dropout_prob)
        self.dropout_5 = nn.Dropout(self.dropout_prob)

        self.batch_norm_1 = nn.BatchNorm2d(self.latent_dim)
        self.batch_norm_2 = nn.BatchNorm2d(self.latent_dim * 2)
        self.batch_norm_3 = nn.BatchNorm2d(self.latent_dim * 4)
        self.batch_norm_4 = nn.BatchNorm2d(self.latent_dim * 8)

        self.channel_mixer_1 = nn.Linear(self.input_channels, self.latent_dim)
        self.channel_mixer_2 = nn.Linear(self.latent_dim, self.latent_dim * 2)
        self.channel_mixer_3 = nn.Linear(self.latent_dim * 2, self.latent_dim * 4)
        self.channel_mixer_4 = nn.Linear(self.latent_dim * 4, self.latent_dim * 8)
        self.channel_mixer_5 = nn.Linear(self.latent_dim * 8, self.output_dim)

        # labels layers

        self.label_embedder = nn.Embedding(num_embeddings = 10, embedding_dim = self.embedding_dim)
        self.linear_1 = nn.Linear(self.embedding_dim, self.input_channels * self.kernel_1 * self.kernel_1)
        self.linear_2 = nn.Linear(self.embedding_dim, self.latent_dim * self.kernel_2 * self.kernel_2)
        self.linear_3 = nn.Linear(self.embedding_dim, (self.latent_dim * 2) * self.kernel_3 * self.kernel_3)
        self.linear_4 = nn.Linear(self.embedding_dim, (self.latent_dim * 4) * self.kernel_4 * self.kernel_4)
        self.linear_5 = nn.Linear(self.embedding_dim, (self.latent_dim * 8) * self.kernel_5 * self.kernel_5)

        self.label_activation_function = nn.Tanh()

    def forward(self, input, label):

        label_embedded = self.label_embedder(label)
        label_weight_layer_1 = self.label_activation_function(
            self.linear_1(label_embedded).view(-1, self.input_channels, self.kernel_1, self.kernel_1))
        label_weight_layer_2 = self.label_activation_function(
            self.linear_2(label_embedded).view(-1, self.latent_dim, self.kernel_2, self.kernel_2))
        label_weight_layer_3 = self.label_activation_function(
            self.linear_3(label_embedded).view(-1, self.latent_dim * 2, self.kernel_3, self.kernel_3))
        label_weight_layer_4 = self.label_activation_function(
            self.linear_4(label_embedded).view(-1, self.latent_dim * 4, self.kernel_4, self.kernel_4))
        label_weight_layer_5 = self.label_activation_function(
            self.linear_5(label_embedded).view(-1, self.latent_dim * 8, self.kernel_5, self.kernel_5))

        # input is 1 x 28 x 28 for MNIST
        output = batch_channel_wise_2d_convolution(
            image = input,
            kernel = label_weight_layer_1,
            stride = 2,
            padding = 1,
            bias = None
        )
        output = self.channel_mixer_1(output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        output = self.dropout_1(output)
        output = self.activation(output)

        # shape is latent_dim x 14 x 14
        output = self.batch_norm_1(output)
        output = batch_channel_wise_2d_convolution(
            image = output,
            kernel = label_weight_layer_2,
            stride = 2,
            padding = 1,
            bias = None
        )
        output = self.channel_mixer_2(output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        output = self.dropout_2(output)
        output = self.activation(output)

        # shape is (latent_dim*2) x 7 * 7
        output = self.batch_norm_2(output)
        output = batch_channel_wise_2d_convolution(
            image = output,
            kernel = label_weight_layer_3,
            stride = 2,
            padding = 1,
            bias = None
        )
        output = self.channel_mixer_3(output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        output = self.dropout_3(output)
        output = self.activation(output)

        # shape is (latent_dim*4) x 4 x 4
        output = self.batch_norm_3(output)
        output = batch_channel_wise_2d_convolution(
            image = output,
            kernel = label_weight_layer_4,
            stride = 2,
            padding = 1,
            bias = None
        )
        
        output = self.channel_mixer_4(output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        output = self.dropout_4(output)
        output = self.activation(output)

        # shape is (latent_dim*8) x 2 x 2
        output = self.batch_norm_4(output)
        output = batch_channel_wise_2d_convolution(
            image = output,
            kernel = label_weight_layer_5,
            stride = 1,
            padding = 0,
            bias = None
        )
        output = self.channel_mixer_5(output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        output = self.dropout_5(output)
        output = self.final_activation(output)

        # shape is output_dim x 1 x 1

        return output, label