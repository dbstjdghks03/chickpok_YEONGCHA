import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PCAModel(nn.Module):
    def __init__(self, n_components):
        super(PCAModel, self).__init__()
        self.Resnet = Resnet()
        self.PCA = PCA(n_components=n_components)
        self.SVM = nn.Linear(2 * n_components, 1)

    def forward(self, mfcc, sc):
        res = self.Resnet(mfcc)
        sc = sc.squeeze()
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
        mean = torch.mean(x, dim=0)
        x = x - mean

        U, S, Vt = torch.linalg.svd(x)
        x @ Vt.t()

        return x @ Vt.t()


if __name__ == "__main__":
    a = torch.rand(16, 3, 200, 3000)
    b = torch.rand(16, 2000, 1)

    pca = PCAModel(10)
    pca(a,b)
