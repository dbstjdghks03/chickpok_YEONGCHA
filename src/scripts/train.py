class Trainer:
    def __init__(self, model, criterion, optimizer, device, data_loader, valid_data_loader, lr_scheduler):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.len_epoch = len(self.data_loader)
    def _train_epoch(self, epoch):
        self.model.train()
        print(f"{epoch}th epoch starting.")
        running_test_loss = 0
        for i, (data, target) in enumerate(train_dataloader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            train_loss = self.criterion(self.model(data), target)
            train_loss.backward()
            self.optimizer.step()

            #for logging

            # self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            # self.train_metrics.update('loss', loss.item())
            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(output, target))

            # if batch_idx % self.log_step == 0:
            #     self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
            #         epoch,
            #         self._progress(batch_idx),
            #         loss.item()))
            #     self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

        # model.eval()
        # running_train_loss = 0.0
        # running_test_loss = 0.0
        # for i, (images, labels) in enumerate(train_dataloader, 0):
        #     images, labels = images.to(device), labels.to(device)
        #     running_train_loss += loss_function(model(images), labels).item() / images.shape[0]
        # for i, (images, labels) in enumerate(test_dataloader, 0):
        #     images, labels = images.to(device), labels.to(device)
        #     running_test_loss += loss_function(model(images), labels).item() / images.shape[0]
        # train_losses.append(running_train_loss)
        # test_losses.append(running_test_loss)


train_dataloader = FERTrainDataLoader(batch_size=10)  # 학습용 데이터셋
test_dataloader = FERTestDataLoader()  # 테스트용 데이터셋
test_dataset = FERTestDataSet()
# 모델 정의한 후 device로 보내기
model = EfficientNet.from_pretrained('EfficientNet-b7', num_classes=7, in_channels=1).to(device)
# loss function 정의 classification 문제이니 CrossEntropyLoss 사용
loss_function = torch.nn.CrossEntropyLoss()
# optimizer로는 Adam 사용
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses = []
test_losses = []
# running_loss = 0.0
for epoch in range(epochs):
    model.train()
    print(f"{epoch}th epoch starting.")
    running_test_loss = 0
    for i, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        train_loss = loss_function(model(images), labels)
        train_loss.backward()

        optimizer.step()

    model.eval()
    running_train_loss = 0.0
    running_test_loss = 0.0
    for i, (images, labels) in enumerate(train_dataloader, 0):
        images, labels = images.to(device), labels.to(device)
        running_train_loss += loss_function(model(images), labels).item() / images.shape[0]
    for i, (images, labels) in enumerate(test_dataloader, 0):
        images, labels = images.to(device), labels.to(device)
        running_test_loss += loss_function(model(images), labels).item() / images.shape[0]
    train_losses.append(running_train_loss)
    test_losses.append(running_test_loss)

test_loss, correct, total = 0, 0, 0
for i, (images, labels) in enumerate(test_dataloader):
    images, labels = images.to(device), labels.to(device)

    output = model(images)
    test_loss += loss_function(output, labels).item()

    pred = output.max(1, keepdim=True)[1]
    correct += pred.eq(labels.view_as(pred)).sum().item()

    total += labels.size(0)

print('[Test set] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss / total, correct, total,
    100. * correct / total))
plt.plot(test_losses, label="test_loss")
plt.plot(train_losses, label="train_loss")
plt.legend()
plt.show()

torch.save(model.state_dict(), 'model.pt')