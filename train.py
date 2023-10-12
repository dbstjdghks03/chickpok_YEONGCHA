import torch
import argparse
from src.datasets.dataset import YoungDataSet
import os
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

# 3. parser.add_argument로 받아들일 인수를 추가해나간다.
parser.add_argument('--epochs', type=int, default=128)
args = parser.parse_args()
root = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = args.epochs
n_components = 256
print(root)
dataset = YoungDataSet(root=root)
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len

train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
#
# if __name__ == '__main__':
#     train_dataloader = TrainDataLoader(root, )
#     test_dataloader = None  # 테스트용 데이터셋
#     test_dataset = FERTestDataSet()
#     # 모델 정의한 후 device로 보내기
#     model = Model(n_components).to(device)
#     # optimizer로는 Adam 사용
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#
#     train_losses = []
#     test_losses = []
#     for epoch in range(epochs):
#         model.train()
#         print(f"{epoch}th epoch starting.")
#         running_test_loss = 0
#         for i, (images, labels) in enumerate(train_dataloader):
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             train_loss = torch.clamp(1 - model(images) * labels, min=0)
#             train_loss.backward()
#             optimizer.step()
#
#     #     model.eval()
#     #     running_train_loss = 0.0
#     #     running_test_loss = 0.0
#     #     for i, (images, labels) in enumerate(train_dataloader, 0):
#     #         images, labels = images.to(device), labels.to(device)
#     #         running_train_loss += loss_function(model(images), labels).item() / images.shape[0]
#     #     for i, (images, labels) in enumerate(test_dataloader, 0):
#     #         images, labels = images.to(device), labels.to(device)
#     #         running_test_loss += loss_function(model(images), labels).item() / images.shape[0]
#     #     train_losses.append(running_train_loss)
#     #     test_losses.append(running_test_loss)
#     #
#     # test_loss, correct, total = 0, 0, 0
#     # for i, (images, labels) in enumerate(test_dataloader):
#     #     images, labels = images.to(device), labels.to(device)
#     #
#     #     output = model(images)
#     #     test_loss += loss_function(output, labels).item()
#     #
#     #     pred = output.max(1, keepdim=True)[1]
#     #     correct += pred.eq(labels.view_as(pred)).sum().item()
#     #
#     #     total += labels.size(0)
#     #
#     # print('[Test set] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
#     #     test_loss / total, correct, total,
#     #     100. * correct / total))
#     # plt.plot(test_losses, label="test_loss")
#     # plt.plot(train_losses, label="train_loss")
#     # plt.legend()
#     # plt.show()
#     #
#     # torch.save(model.state_dict(), 'model.pt')
