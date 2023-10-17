import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.decomposition import PCA


class PCAModel(nn.Module):
    def __init__(self, n_components):
        super(PCAModel, self).__init__()
        self.Resnet = Resnet()
        self.PCA = PCA(n_components=n_components)
        self.SVM = nn.Linear(2 * n_components, 1)

    def forward(self, mfcc, sc):
        res = self.Resnet(mfcc)
        res_reduced = self.PCA(res)
        sc_reduced = self.PCA(sc)

        print(res_reduced.shape, sc_reduced.shape)

        combined_feat = torch.concat((res_reduced, sc_reduced), -1)
        out = self.SVM(combined_feat)

        return out


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(resnet50.children())[:-1])

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        return x.view(x.size(0), -1)


class PCA(nn.Module):
    def __init__(self, n_components):
        super(PCA, self).__init__()
        self.n_components = n_components

    def forward(self, x):
        x = x - x.mean(dim=0)
        U, S, V = torch.svd(x)
        PC = V[:, :self.n_components]

        return x @ PC

