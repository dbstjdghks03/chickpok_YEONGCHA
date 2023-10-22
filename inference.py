import torch
import numpy as np
import re
from nptdms import TdmsFile
import argparse

from src.datasets.dataset import YoungDataSet, PreProcess, TestYoungDataSet
from src.models.PCAmodel import PCAModel

parser = argparse.ArgumentParser()

# 3. parser.add_argument로 받아들일 인수를 추가해나간다.
parser.add_argument('--root', type=str, default="dataset")

args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_components = 40

if __name__ == '__main__':
    model = PCAModel(n_components).to(device)
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    position_mse = 0
    test_len = 0
    correct_predictions = 0

    test_dataset = YoungDataSet(root=root, is_npy=True, transform=None)

    for i, (mfcc, sc, horn, position) in enumerate(test_loader):
        mfcc, sc, horn, position = mfcc.to(device).float(), sc.to(
            device).float(), horn.to(device), position.to(device).float()
        output = model(mfcc, sc)
        epoch_test_loss += loss(output, horn, position).item()

        predictions = torch.tensor([1 if out[0] > 0 else -1 for out in output]).to(device)
        correct_predictions += (predictions == horn).float().sum()
        position_mse += MSELoss(output[:, 1], position)
        test_len += position.shape[0]

    accuracy = correct_predictions / test_len

    print('Horn Accuracy: {}/{} ({:.2f}%), Position MSE: {}\n'.format(
        correct_predictions, test_len,
        accuracy, position_mse / test_len))



