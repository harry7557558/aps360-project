import torch
import torch.nn as nn

def conv3s(n_in, n_out, stride):
    return nn.Conv2d(n_in, n_out, 3, stride=stride, padding=1)

class DownscaleBlock(nn.Module):
    def __init__(self, n, slope):
        super(DownscaleBlock, self).__init__()
        self.conv1 = conv3s(n, n, 2)
        self.bn1 = nn.BatchNorm2d(n)
        self.lrelu1 = nn.LeakyReLU(slope, inplace=True)
        
        self.conv2 = conv3s(n, n, 1)
        self.bn2 = nn.BatchNorm2d(n)
        self.lrelu2 = nn.LeakyReLU(slope, inplace=True)
        
    def forward(self, x):
        x = self.lrelu1(self.bn1(self.conv1(x)))
        return self.lrelu2(self.bn2(self.conv2(x)))

class DiscriminatorModel(nn.Module):
    def __init__(self, n_in, n_out, n_hidden, n_downscale, slope):
        super(DiscriminatorModel, self).__init()
        self.convi = conv3s(n_in, n_hidden, 1)
        self.lrelu1 = nn.LeakyReLU(slope, inplace=True)
        
        self.downscale = nn.Sequential(
            *[DownscaleBlock(n_hidden) for _ in range(n_downscale)]
        )
        
        self.ampl = nn.AdaptiveMaxPool2d((4,4))
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(n_hidden, n_hidden)
        self.lrelu2 = nn.LeakyReLU(slope, inplace=True)
        self.dense2 = nn.Linear(n_hidden, n_hidden)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        xi = self.lrelu1(self.convi(x))
        xd = self.downscale(xi)
        xf = self.flatten(self.ampl(xd))
        xc1 = self.lrelu2(self.dense1(xf))
        xc2 = self.sigmoid(self.dense2(xc1))
        return xc2
