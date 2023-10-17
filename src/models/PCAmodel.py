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
        res_reduced = self.PCA.transform(res)
        sc_reduced = self.PCA.transform(sc)

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
