import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class CustomBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            ConvBlock(3, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 256),
            nn.MaxPool2d(2),
            ConvBlock(256, 512),
        )
        self.out_channels = 512

    def forward(self, x):
        return self.body(x)
