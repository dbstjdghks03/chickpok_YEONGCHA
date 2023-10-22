import lightning.pytorch as L
from collections import defaultdict
import os
import numpy as np
from pathlib import Path
import copy
import torch.nn as nn
import time
import torch
from PCAmodel import PCAModel


class PCALightModel(L.LightningModule):
    def __init__(self, n_components, num_epochs, lr, loss, alpha, beta):
        super(self, PCALightModel).__init__()
        self.lr = lr
        self.loss = loss
        self.report_interval = report_interval
        self.num_epochs = num_epochs
        self.val_start = 0
        self.model = PCAModel(n_components)
        self.alpha = alpha
        self.beta = beta
        self.start_epoch = 0
        self.start_iter = 0
        self.len_loader = 0

        self.total_loss = 0

        self.val_loss_log = np.empty((0, 2))

        self.best_val_loss = 1.

        self.train_time = 0

    def on_train_start(self):
        print('Current cuda device: ', self.device)

    def on_train_epoch_start(self):
        print(f'Epoch #{self.current_epoch:2d} ............... {self.net_name} ...............')

        self.start_epoch = self.start_iter = time.perf_counter()
        self.total_loss = 0.

    def training_step(self, batch, batch_idx):
        mfcc, sc, horn, position = batch
        output = model(mfcc, sc)
        train_loss = self.loss(output, horn, position, self.alpha, self.beta)
        self.log("train_loss", train_loss, on_epoch=True, prog_bar=True)

        return train_loss

    # def on_train_epoch_end(self):
    #     self.total_loss = torch.tensor(self.total_loss / self.trainer.num_training_batches)
    #     self.train_time = time.perf_counter() - self.start_epoch

    # def on_validation_epoch_start(self):
    #     self.val_start = time.perf_counter()

    def validation_step(self, batch, batch_idx):
        mfcc, sc, horn, position = batch
        output = model(mfcc, sc)
        test_loss = self.loss(output, horn, position, self.alpha, self.beta)
        self.log("test_loss", test_loss, on_epoch=True, prog_bar=True)

    # def on_validation_epoch_end(self):
    #     val_loss = sum([ssim_loss(self.targets[fname], self.reconstructions[fname]) for fname in self.reconstructions])
    #     num_subjects = len(self.reconstructions)
    #     val_time = time.perf_counter() - self.val_start
    #
    #     self.val_loss_log = np.append(self.val_loss_log, np.array([[self.current_epoch, val_loss]]), axis=0)
    #     file_path = os.path.join(self.args.val_loss_dir, "val_loss_log")
    #     np.save(file_path, self.val_loss_log)
    #     print(f"loss file saved! {file_path}")
    #
    #     val_loss = torch.tensor(val_loss) / torch.tensor(num_subjects)
    #     self.log("val_loss", val_loss)
    #
    #     is_new_best = val_loss < self.best_val_loss
    #     self.best_val_loss = min(self.best_val_loss, val_loss)
    #
    #     save_model(self.args, self.args.exp_dir, self.current_epoch + 1, self.model, self.configure_optimizers(),
    #                self.best_val_loss, is_new_best)
    #     print(f'Epoch = [{self.current_epoch:4d}/{self.args.num_epochs:4d}] TrainLoss = {self.total_loss:.4g} '
    #           f'ValLoss = {val_loss:.4g} TrainTime = {self.train_time:.4f}s ValTime = {val_time:.4f}s',
    #           )
    #
    #     if is_new_best:
    #         print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    #         start = time.perf_counter()
    #         save_reconstructions(self.reconstructions, self.args.val_dir, targets=self.targets, inputs=None)
    #         print(
    #             f'ForwardTime = {time.perf_counter() - start:.4f}s',
    #         )

    # def on_test_start(self):
    #     print('Current cuda device: ', self.device)
    #     self.test_reconstructions = defaultdict(dict)
    #
    # def test_step(self, batch, batch_idx):
    #     mask, kspace, _, _, fnames, slices = batch
    #     output = self.model(kspace, mask)
    #
    #     for i in range(output.shape[0]):
    #         self.test_reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
