import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.decomposition import PCA


class Model(nn.Module):
    def __init__(self, n_components):
        super(Model, self).__init__()
        self.Resnet = Resnet()
        self.PCA = PCA(n_components=n_components)
        self.fc = nn.Linear(2 * n_components, 1)

    def forward(self, x):
        mfcc = x.mfcc
        sc = x.sc
        res = self.Resnet(mfcc)
        res_reduced = self.PCA(res)
        sc_reduced = self.PCA(sc)

        combined_feat = torch.concat((res_reduced, sc_reduced), -1)
        out = self.fc(combined_feat)

        return out

class SVM(nn.Module):
    def __init__(self, n_components):
        super(SVM, self).__init__()
        self.fc = nn.Linear(2 * n_components, 1)

    def forward(self, x):
        out = self.fc(x)


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        del resnet50.fc
        self.model = resnet50

        for param in model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)
