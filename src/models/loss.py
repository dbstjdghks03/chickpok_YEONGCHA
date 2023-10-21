import torch
from torch import nn


def loss(output, horn, position, alpha=1e-3, beta=1):
    mse_loss = nn.MSELoss()
    mask = (horn == -1).float()

    train_loss = alpha * torch.mean(torch.clamp(1 - output[:, 0] * horn, min=0)) + beta * mse_loss(output[:, 1] * mask,
                                                                                                   position * mask)

    return train_loss


def horn_loss(output, horn, position):
    train_loss = torch.mean(torch.clamp(1 - output[:, 0] * horn, min=0))

    return train_loss
