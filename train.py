import torch
import argparse
from src.datasets.dataset import YoungDataSet
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from src.models.PCAmodel import PCAModel
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser()

# 3. parser.add_argument로 받아들일 인수를 추가해나간다.
parser.add_argument('--epochs', type=int, default=128)
parser.add_argument('--root', type=str)
parser.add_argument('--batch', type=int, default=2)

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root = args.root
epochs = args.epochs
n_components = 20
print(root)
dataset = YoungDataSet(root=root)

# train_len = int(0.8 * len(dataset))
# val_len = len(dataset) - train_len
# print(train_len, val_len)

data_list = dataset.data_list

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_indices, test_indices) in enumerate(skf.split(data_list, [item[5] for item in data_list])):
    train_set = torch.utils.data.Subset(dataset, train_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)

    batch_size = 16
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

if __name__ == '__main__':
    model = PCAModel(n_components).to(device)
    # optimizer로는 Adam 사용
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        model.train()
        print(f"{epoch}th epoch starting.")
        running_test_loss = 0
        for i, (stft, mfcc, sc, horn, position) in enumerate(train_loader):
            stft, mfcc, sc, horn = stft.to(device).float(), mfcc.to(device).float(), sc.to(device).float(), horn.to(device)
            optimizer.zero_grad()
            train_loss = torch.clamp(1 - model(mfcc, sc) * horn, min=0)
            train_loss.backward()
            optimizer.step()

    #     model.eval()
    #     running_train_loss = 0.0
    #     running_test_loss = 0.0
    #     for i, (images, labels) in enumerate(train_dataloader, 0):
    #         images, labels = images.to(device), labels.to(device)
    #         running_train_loss += loss_function(model(images), labels).item() / images.shape[0]
    #     for i, (images, labels) in enumerate(test_dataloader, 0):
    #         images, labels = images.to(device), labels.to(device)
    #         running_test_loss += loss_function(model(images), labels).item() / images.shape[0]
    #     train_losses.append(running_train_loss)
    #     test_losses.append(running_test_loss)
    #
    # test_loss, correct, total = 0, 0, 0
    # for i, (images, labels) in enumerate(test_dataloader):
    #     images, labels = images.to(device), labels.to(device)
    #
    #     output = model(images)
    #     test_loss += loss_function(output, labels).item()
    #
    #     pred = output.max(1, keepdim=True)[1]
    #     correct += pred.eq(labels.view_as(pred)).sum().item()
    #
    #     total += labels.size(0)
    #
    # print('[Test set] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     test_loss / total, correct, total,
    #     100. * correct / total))
    # plt.plot(test_losses, label="test_loss")
    # plt.plot(train_losses, label="train_loss")
    # plt.legend()
    # plt.show()
    #
    # torch.save(model.state_dict(), 'model.pt')
