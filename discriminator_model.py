import torch
import torch.nn as nn

def conv3s(nin, nout, stride):
    return nn.Conv2d(nin, nout, 3, stride=stride, padding=1)

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
