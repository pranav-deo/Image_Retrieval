import torch.nn as nn
import torchvision.models as models


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class AE(nn.Module):
    """docstring for AE"""

    def __init__(self, K):
        super(AE, self).__init__()

        self.pretrained_net = models.resnet50(pretrained=True)
        self.pretrained_net.fc = nn.Sequential(
            nn.BatchNorm1d(2048, affine=False),
            nn.Dropout(p=0.2),
            nn.Linear(2048, K, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(K, affine=False),
            nn.Dropout(p=0.2),
            nn.Linear(K, K, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pretrained_net(x)
