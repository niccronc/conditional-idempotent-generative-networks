'''
The difference between this file and the original one is that here:
1. we explicitly set the model copy's parameters to not require gradients, to speed up the backward pass.
'''

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import L1Loss

import pytorch_lightning as pl

from copy import deepcopy

class IdempotentNetworkWithConditioning(pl.LightningModule):
    def __init__(
        self,
        prior,
        model,
        lr=1e-4,
        criterion=L1Loss(),
        lrec_w=20.0, #reconstruction loss weight
        lidem_noise_w=20.0, #idempotent loss weight for noisy samples
        ltight_noise_w=2.5, #tightness loss weight for noisy samples
        lidem_mismatch_w=3.0, #idempotent loss weight for mismatched samples
        ltight_mismatch_w=1.0, #tightness loss weight for mismatched samples
    ):
        super(IdempotentNetworkWithConditioning, self).__init__()
        self.prior = prior
        self.model = model
        self.model_copy = deepcopy(model)
        for param in self.model_copy.parameters():
            param.requires_grad = False
        
        self.lr = lr
        self.criterion = criterion
        self.lrec_w = lrec_w
        self.lidem_noise_w = lidem_noise_w
        self.ltight_noise_w = ltight_noise_w
        self.lidem_mismatch_w = lidem_mismatch_w
        self.ltight_mismatch_w = ltight_mismatch_w

    def forward(self, x, y):
        return self.model(x, y)

    def configure_optimizers(self):
        optim = AdamW(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return optim

    def get_losses(self, x, y):

        # Updating the copy
        self.model_copy.load_state_dict(self.model.state_dict())

        # Prior samples
        x_noise = self.prior.sample_n(x.shape[0]).to(x.device)
        y_noise = torch.randint(0, 10, (x.shape[0],)).to(x.device)

        # mismatch samples
        x_mismatch, y_mismatch = self._create_additional_pairs_batch(x, y) # casted to x.device within the helper function

        # Forward passes
        fx, _ = self(x, y)
        fnoise, _ = self(x_noise, y_noise)
        fnoise_d = fnoise.detach()
        fmismatch, _ = self(x_mismatch, y_mismatch)
        fmismatch_d = fmismatch.detach()


        l_rec = self.lrec_w * self.criterion(fx, x)
        l_noise_idem = self.lidem_noise_w * self.criterion(self.model_copy(fnoise, y_noise)[0], fnoise)
        l_noise_tight = -self.ltight_noise_w * self.criterion(self(fnoise_d, y_noise)[0], fnoise_d)
        l_mismatch_idem = self.lidem_mismatch_w * self.criterion(self.model_copy(fmismatch, y_mismatch)[0], fmismatch)
        l_mismatch_tight = -self.ltight_mismatch_w * self.criterion(self(fmismatch_d, y_mismatch)[0], fmismatch_d)

        return l_rec, l_noise_idem, l_noise_tight, l_mismatch_idem, l_mismatch_tight

    def training_step(self, batch, batch_idx):
        x, y = batch

        l_rec, l_noise_idem, l_noise_tight, l_mismatch_idem, l_mismatch_tight = self.get_losses(x, y)
        loss = l_rec + l_noise_idem + l_noise_tight + l_mismatch_idem + l_mismatch_tight

        self.log_dict(
            {
                "train/loss_rec": l_rec,
                "train/loss_idem_noise": l_noise_idem,
                "train/loss_tight_noise": l_noise_tight,
                "train/loss_idem_mismatch": l_mismatch_idem,
                "train/loss_tight_mismatch": l_mismatch_tight,
                "train/loss": loss,
            },
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        l_rec, l_noise_idem, l_noise_tight, l_mismatch_idem, l_mismatch_tight = self.get_losses(x, y)
        loss = l_rec + l_noise_idem + l_noise_tight + l_mismatch_idem + l_mismatch_tight

        self.log_dict(
            {
                "val/loss_rec": l_rec,
                "val/loss_idem_noise": l_noise_idem,
                "val/loss_tight_noise": l_noise_tight,
                "val/loss_idem_mismatch": l_mismatch_idem,
                "val/loss_tight_mismatch": l_mismatch_tight,
                "val/loss": loss,
            },
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch

        l_rec, l_noise_idem, l_noise_tight, l_mismatch_idem, l_mismatch_tight = self.get_losses(x, y)
        loss = l_rec + l_noise_idem + l_noise_tight + l_mismatch_idem + l_mismatch_tight

        self.log_dict(
            {
                "test/loss_rec": l_rec,
                "test/loss_idem_noise": l_noise_idem,
                "test/loss_tight_noise": l_noise_tight,
                "test/loss_idem_mismatch": l_mismatch_idem,
                "test/loss_tight_mismatch": l_mismatch_tight,
                "test/loss": loss,
            },
            sync_dist=True,
        )

    def generate_n(self, n, label, device=None):
        z = self.prior.sample_n(n)
        y = torch.full((n,), label)

        if device is not None:
            z = z.to(device)

        return self(z, y)[0]

    def _create_additional_pairs_batch(self, batch_x, batch_y):
        """
        Creates additional pairs (x, y') with varying labels y' and outputs tensors.

        Args:
          batch_x: A tensor of shape (B, ...) representing the batch of images.
          batch_y: A tensor of shape (B,) representing the corresponding labels.

        Returns:
          x_new: A tensor of shape (9*B, ...) containing repeated images with varying labels.
          y_new: A tensor of shape (9*B,) containing the new labels.
        """

        N = batch_x.size(0)  # Get batch size
        x_new = torch.empty((9 * N,) + batch_x.shape[1:])  # Pre-allocate for efficiency
        y_new = torch.empty((9 * N,))

        idx = 0
        for i in range(N):
            x = batch_x[i]
            y = batch_y[i]
            labels = list(range(10))
            labels.remove(y)

            for label in labels:
                x_new[idx] = x.clone()  # Create a copy of x for independence
                y_new[idx] = label
                idx += 1

        return x_new.to(x.device), y_new.long().to(x.device)