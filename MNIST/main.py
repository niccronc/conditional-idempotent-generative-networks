import os
from argparse import ArgumentParser
from time import time
import matplotlib.pyplot as plt
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from model_with_channel_conditioning import DCGANLikeModelWithChannelConditioning
from idempotent_networks_v2 import IdempotentNetworkWithConditioning


def argument_parser():
    parser = ArgumentParser(description='Parse training arguments')
    parser.add_argument('-s', '--seed', type=int, help='Random seed', default=0)
    parser.add_argument('-ld', '--latent-dim', type=int, help='Latent dimension', default=16)
    parser.add_argument('-id', '--intermediate-dim', type=int, help='Intermediate dimension', default=512)
    parser.add_argument('-ed', '--embedding-dim', type=int, help='Embedding dimension', default=10)
    parser.add_argument('-lr', '--learning-rate', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('-b', '--batch-size', type=int, help='Batch size', default=256)
    parser.add_argument('-e', '--num-epochs', type=int, help='Number of training epochs', default=100)
    parser.add_argument('-w', '--num-workers', type=int, help='Number of workers to use during training', default=0)
    parser.add_argument('-d', '--download', help='Download the dataset from source', action='store_true')
    return parser.parse_args()

def load_data(args):

    # transformations
    normalization = Lambda(lambda x: (x - 0.5) * 2)
    noise = Lambda(lambda x: (x + torch.randn_like(x) * 0.15).clamp(-1, 1))
    train_transform = Compose([ToTensor(), normalization, noise])
    val_transform = Compose([ToTensor(), normalization])

    # datasets
    train_dataset = MNIST(
        root="mnist", 
        train=True, 
        download=args.download, 
        transform=train_transform
    )
    val_dataset = MNIST(
        root="mnist", 
        train=False, 
        download=args.download,
        transform=val_transform
    )

    #dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    return train_loader, val_loader

def initialize_model(args):
    
    # Initialize model
    prior = torch.distributions.Normal(torch.zeros(1, args.image_size, args.image_size), 
                                       torch.ones(1, args.image_size, args.image_size))
    net = DCGANLikeModelWithChannelConditioning(data_channels=1, 
                                                latent_dim=args.latent_dim, 
                                                intermediate_dim=args.intermediate_dim, 
                                                embedding_dim=args.embedding_dim)
    model = IdempotentNetworkWithConditioning(prior, net, args.learning_rate)
    
    return model

def train_model(args, model, train_dataloader, val_dataloader):

    logging_timestamp = int(time())
    print(f'timestamp used for logging is {logging_timestamp}')

    callbacks = [
      ModelCheckpoint(
          monitor="val/loss",
          mode="min",
          dirpath=f"checkpoints_with_channel_conditioning/{logging_timestamp}",
          filename="{epoch}_best_checkpoint",
          every_n_epochs = 5,
          verbose = True,
      )
    ]

    trainer = Trainer(
      strategy="ddp",
      accelerator="auto",
      max_epochs=args.num_epochs,
      logger=TensorBoardLogger(f"channel_conditioning/{logging_timestamp}", 
                               name=f"DCGAN_with_channel_conditioning_{logging_timestamp}"),
      callbacks=callbacks,
    )

    print('starting to train the model')
    trainer.fit(model, train_dataloader, val_dataloader)
    
def main():
    args = argument_parser()
    args.image_size = 28 # MNIST size
    print(f'parsed arguments as {args}')
    
    pl.seed_everything(args.seed)
    
    train_loader, val_loader = load_data(args)
    print(f'loaded data')
    
    model = initialize_model(args)
    print(f'initialized model')
    
    train_model(args, model, train_loader, val_loader)

if __name__ == '__main__':
    main()