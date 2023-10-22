import torch
import argparse
from src.datasets.dataset import YoungDataSet
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from src.models.PCAmodel import PCAModel
from sklearn.model_selection import StratifiedKFold
from src.models.loss import loss, horn_loss
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import random
import numpy as np

seed = 42
# loss = horn_loss

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()

# 3. parser.add_argument로 받아들일 인수를 추가해나간다.
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--root', type=str)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=os.cpu_count())
parser.add_argument('--n_components', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--alpha', type=float, default=1e-3)
parser.add_argument('--beta', type=float, default=1)

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root = args.root
epochs = args.epochs
batch = args.batch
num_workers = args.num_workers
n_components = args.n_components
alpha = args.alpha
beta = args.beta
lr = args.lr

if __name__ == '__main__':
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    MSELoss = nn.MSELoss(reduction="sum")

    transform = ["amp", "flip", "neg", "awgn", "abgn", "argn", "avgn", "apgn", "sine", "ampsegment", "aun", "phn",
                 "fshift"]
    model = PCAModel(n_components).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dataset = YoungDataSet(root=root, is_npy=True, transform=transform)
    test_dataset = YoungDataSet(root=root, is_npy=True, transform=None)
    data_list = train_dataset.data_list
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        print(f"{epoch}th epoch starting.")
        for fold, (train_indices, test_indices) in enumerate(skf.split(data_list, [item[5] for item in data_list])):
            print(f"fold {fold} starting")
            train_set = torch.utils.data.Subset(train_dataset, train_indices)
            test_set = torch.utils.data.Subset(test_dataset, test_indices)

            train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True)
            test_loader = DataLoader(test_set, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)


            # optimizer로는 Adam 사용

            model.train()
            epoch_test_loss = 0
            epoch_train_loss = 0
            train_len = 0
            for i, (mfcc, sc, horn, position) in enumerate(train_loader):
                mfcc, sc, horn, position = mfcc.to(device).float(), sc.to(
                    device).float(), horn.to(device), position.to(device).float()
                optimizer.zero_grad()
                output = model(mfcc, sc)
                train_loss = loss(output, horn, position, alpha, beta)
                epoch_train_loss += train_loss.item()
                train_loss.backward()
                optimizer.step()
                train_len += position.shape[0]

            model.eval()
            position_mse = 0
            test_len = 0
            correct_predictions = 0
            lst_for_var = []

            for i, (mfcc, sc, horn, position) in enumerate(test_loader):
                mfcc, sc, horn, position = mfcc.to(device).float(), sc.to(
                    device).float(), horn.to(device), position.to(device).float()
                output = model(mfcc, sc)
                epoch_test_loss += loss(output, horn, position).item()

                predictions = torch.tensor([1 if out[0] > 0 else -1 for out in output]).to(device)
                print(predictions, horn, predictions == horn)
                correct_predictions += (predictions == horn).float().sum()
                position_mse += MSELoss(output[:, 1], position)
                lst_for_var.append(MSELoss(output[:, 1], position)**2)

                test_len += position.shape[0]

            accuracy = correct_predictions / test_len
            std = torch.mean(torch.tensor(lst_for_var))**(1/2)
            confi_interval_width = 3.92 * std / (len(torch.tensor(lst_for_var))**(1/2))

            print(f'train_loss: {epoch_train_loss/train_len}')
            print('[Test set] Average loss: {:.4f}, Horn Accuracy: {}/{} ({:.2f}%), Confidence Interval Width: {}\n'.format(
                epoch_test_loss / test_len, correct_predictions, test_len,
                accuracy, confi_interval_width))

            train_losses.append(epoch_train_loss)
            test_losses.append(epoch_test_loss)

        torch.save(model.state_dict(), f'{epoch}_model.pt')

    plt.plot(test_losses, label="test_loss")
    plt.plot(train_losses, label="train_loss")
    plt.legend()
    plt.show()
