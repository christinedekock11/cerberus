import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .datautils import SparseDataset, Reader
from .losses import WMSE
from .cerberus import MatFact

class DataModulePL(pl.LightningDataModule):
    def __init__(self, train_path_A, train_path_C, batch_size, val_frac=0.1):
        super().__init__()
        self.train_path_A = train_path_A
        self.train_path_C = train_path_C
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.shape_A = []
        self.shape_C = []
        self.T = 0

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Load the data for A and C separately
            mats, train_inds, val_inds = Reader(self.val_frac).run(self.train_path_A)
            self.train_ds_A = SparseDataset(mats, train_inds, 'A')
            self.val_ds_A = SparseDataset(mats, val_inds, 'A')
            self.shape_A = mats[0].shape

            mats, train_inds, val_inds = Reader(self.val_frac).run(self.train_path_C)
            self.train_ds_C = SparseDataset(mats, train_inds, 'C')
            self.val_ds_C = SparseDataset(mats, val_inds, 'C')
            self.shape_C = mats[0].shape
            self.T = len(mats)

    def train_dataloader(self):
        # Create separate dataloaders for A and C
        train_dl_A = DataLoader(self.train_ds_A, batch_size=self.batch_size, num_workers=2, persistent_workers=True)
        train_dl_C = DataLoader(self.train_ds_C, batch_size=self.batch_size, num_workers=2, persistent_workers=True)
        return train_dl_A, train_dl_C

    def val_dataloader(self):
        # Create separate dataloaders for A and C
        val_dl_A = DataLoader(self.val_ds_A, batch_size=self.batch_size, num_workers=2, persistent_workers=True)
        val_dl_C = DataLoader(self.val_ds_C, batch_size=self.batch_size, num_workers=2, persistent_workers=True)
        return val_dl_A, val_dl_C

class MatFactPL(pl.LightningModule):
    def __init__(self, **hyperparameters):
        super().__init__()
        self.save_hyperparameters()
        print('**** Hyperparams: ****')
        _ = [print(k,v) for k,v in self.hparams.items()]
        print('**********************')

        self.MatFact = MatFact(N=self.hparams.N, M=self.hparams.M,
                    D=self.hparams.D, K=self.hparams.K, T=self.hparams.T)

        self.loss_fn = WMSE(self.hparams.c0_scaler)
        self.validation_step_outputs = []

    def forward(self, t, user_ind, source):
        return self.MatFact(t, user_ind, source)

    def training_step(self, batches, batch_idx):
        losses = {'A': 0, 'C': 0}

        for batch in batches: # batches = 2 dataloaders
            t, user_ind, target, source = batch
            preds = self(t, user_ind, source)
            losses[source[0]] += self.loss_fn(preds, target)

        # Total loss = weighted loss for both sources + regularization
        loss = self.hparams.weight_A * losses['A'] + self.hparams.weight_C * losses['C'] + \
               self.hparams.lambda_1 * self.MatFact.weight_regularization() + \
               self.hparams.lambda_2 * self.MatFact.alignment_regularization()

        self.log("train_loss", loss, sync_dist=True, prog_bar=True, logger=True,
                 on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        t, user_ind, target, source = batch
        preds = self(t, user_ind, source)
        loss = self.loss_fn(preds, target)

        # Return loss inside a dictionary
        self.validation_step_outputs.append((source[0],loss))
        return loss

    def on_validation_epoch_end(self):
        # Aggregate the losses from each dataloader
        losses = {'A':[],'C':[]}
        for source,loss in self.validation_step_outputs:
            losses[source].append(loss)

        # Combine the losses with regularization
        total_loss = self.hparams.weight_A * sum(losses['A'])/len(losses['A']) + \
                      self.hparams.weight_C * sum(losses['C'])/len(losses['C']) + \
                      self.hparams.lambda_1 * self.MatFact.weight_regularization() + \
                      self.hparams.lambda_2 * self.MatFact.alignment_regularization()

        # Return the final loss
        self.log("val_loss", total_loss, sync_dist=True, prog_bar=True, logger=True,
                 on_step=False, on_epoch=True)
        self.validation_step_outputs.clear()
        return {'val_loss': total_loss}

        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return opt