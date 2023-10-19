import torch
from torch import nn


def loss(output, horn, position):
    mse_loss = nn.MSELoss()
    train_loss = torch.mean(torch.clamp(1 - output[:, 0] * horn, min=0)) + mse_loss(output[:, 1], position)

    return train_loss
