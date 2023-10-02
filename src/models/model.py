import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.decomposition import PCA
from sklearn.svm import SVC

class Model(nn.Module):
    def __init__(self, time, selected_feature, n_components):
        super(Model, self).__init__()
        self.Resnet = Resnet()
        self.PCA = PCA(n_components=n_components)
        self.fc = nn.Linear(2 * n_components, 1)
        self.input = selected_feature

    def forward(self, processed_data):
        res = self.Resnet(input)
        sc = processed_data.sc
        res_reduced = self.PCA(res)
        sc_reduced = self.PCA(sc)

        combined_feat = torch.concat((res_reduced, sc_reduced), -1)
        out = self.fc(combined_feat)

        return out

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        del resnet50.fc
        self.model = resnet50

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)

class SVM(nn.Module):
    def __init__(self, n_components):
        super(SVM, self).__init__()
        self.fc = nn.Linear(2 * n_components, 1)

    def forward(self, x):
        out = self.fc(x)
        return out