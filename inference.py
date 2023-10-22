import torch
import numpy as np
import re
from nptdms import TdmsFile
import argparse

from src.datasets.dataset import YoungDataSet, PreProcess, TestYoungDataSet
from src.models.PCAmodel import PCAModel
import pandas as pd
from torch.utils.data import DataLoader
import os
parser = argparse.ArgumentParser()

# 3. parser.add_argument로 받아들일 인수를 추가해나간다.
parser.add_argument('--root', type=str, default="dataset")
parser.add_argument('--num_workers', type=int, default=os.cpu_count())

args = parser.parse_args()

root = args.root
num_workers = args.num_workers
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_components = 40

if __name__ == '__main__':
    model = PCAModel(n_components).to(device)
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    position_mse = 0
    test_len = 0
    correct_predictions = 0

    test_dataset = TestYoungDataSet(root=root, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)

    df = pd.DataFrame(columns=['Predicted_Danger', 'Predicted_Position', 'label_Horn', 'label_Position'])

    for i, (mfcc, sc, horn, position) in enumerate(test_loader):
        mfcc, sc, horn, position = mfcc.to(device).float(), sc.to(
            device).float(), horn.to(device), position.to(device).float()
        output = model(mfcc, sc)
        epoch_test_loss += loss(output, horn, position).item()

        danger = output[:, 0]
        predicted_position = output[:, 1]

        predictions = torch.tensor([1 if out[0] > 0 else -1 for out in output]).to(device)
        correct_predictions += (predictions == horn).float().sum()
        position_mse += MSELoss(output[:, 1], position)
        test_len += position.shape[0]
        batch_data = {
            'Predicted_Danger': danger.detach().numpy(),
            'Predicted_Position': predicted_position.detach().numpy(),
            'Output': output[:, 0].cpu().detach().numpy(),  # assuming the first dimension of output is what you want
            'Horn': horn.cpu().numpy()
        }
        df = df.append(pd.DataFrame(batch_data), ignore_index=True)

    accuracy = correct_predictions / test_len

    print('Horn Accuracy: {}/{} ({:.2f}%), Position MSE: {}\n'.format(
        correct_predictions, test_len,
        accuracy, position_mse / test_len))

    df.to_csv('result.csv')
