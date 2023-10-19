import torch
import argparse
from src.datasets.dataset import YoungDataSet
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from src.models.PCAmodel import PCAModel
from sklearn.model_selection import StratifiedKFold
from src.models.loss import loss
import matplotlib.pyplot as plt
import torch.nn as nn
import os

parser = argparse.ArgumentParser()

# 3. parser.add_argument로 받아들일 인수를 추가해나간다.
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--root', type=str)
parser.add_argument('--batch', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=os.cpu_count())
parser.add_argument('--n_components', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-5)

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root = args.root
epochs = args.epochs
batch = args.batch
num_workers = args.num_workers
n_components = args.n_components
lr = args.lr

dataset = YoungDataSet(root=root, is_npy=True)
data_list = dataset.data_list

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
MSELoss = nn.MSELoss()
if __name__ == '__main__':
    model = PCAModel(n_components).to(device)
    # optimizer로는 Adam 사용
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    valid_best = 99999999
    for fold, (train_indices, test_indices) in enumerate(skf.split(data_list, [item[5] for item in data_list])):
        train_set = torch.utils.data.Subset(dataset, train_indices)
        test_set = torch.utils.data.Subset(dataset, test_indices)
        train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_set, batch_size=batch, shuffle=False, num_workers=num_workers)

        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            model.train()
            print(f"font {fold}: {epoch}th epoch starting.")
            epoch_test_loss = 0
            epoch_train_loss = 0
            for i, (stft, mfcc, sc, horn, position) in enumerate(train_loader):
                stft, mfcc, sc, horn, position = stft.to(device).float(), mfcc.to(device).float(), sc.to(
                    device).float(), horn.to(device), position.to(device).float()
                optimizer.zero_grad()
                output = model(mfcc, sc)
                train_loss = loss(output, horn, position)
                epoch_train_loss += train_loss.item()
                train_loss.backward()
                optimizer.step()

            model.eval()
            position_mse = 0
            test_len = 0
            for i, (stft, mfcc, sc, horn, position) in enumerate(test_loader):
                stft, mfcc, sc, horn, position = stft.to(device).float(), mfcc.to(device).float(), sc.to(
                    device).float(), horn.to(device), position.to(device).float()
                output = model(mfcc, sc)
                epoch_test_loss += loss(output, horn, position).item()

                predictions = torch.tensor([1 if 1 - out[0] > 0 else -1 for out in output]).to(device)
                correct_predictions = (predictions == horn).float().sum()
                position_mse += MSELoss(output[:, 1], position)
                test_len += position.shape[0]

            accuracy = correct_predictions / test_len

            print(f'train_loss: {epoch_train_loss}')
            print('[Test set] Average loss: {:.4f}, Horn Accuracy: {}/{} ({:.2f}%), Position MSE: {}\n'.format(
                epoch_test_loss / len(test_loader), correct_predictions, test_len,
                accuracy, position_mse))

            train_losses.append(epoch_train_loss)
            test_losses.append(epoch_test_loss)
            if valid_best > epoch_test_loss:
                valid_best = epoch_test_loss
                torch.save(model.state_dict(), 'valid_best_model.pt')

    plt.plot(test_losses, label="test_loss")
    plt.plot(train_losses, label="train_loss")
    plt.legend()
    plt.show()

