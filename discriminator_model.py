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
        
        self.conv2 = conv3s(n, 2*n, 1)
        self.bn2 = nn.BatchNorm2d(2*n)
        self.lrelu2 = nn.LeakyReLU(slope, inplace=True)
        
    def forward(self, x):
        x = self.lrelu1(self.bn1(self.conv1(x)))
        return self.lrelu2(self.bn2(self.conv2(x)))

class DiscriminatorModel(nn.Module):
    def __init__(self, n_in, n_hidden, slope):
        super(DiscriminatorModel, self).__init__()
        self.proc_input = nn.Sequential(
            conv3s(n_in, n_hidden, 1),
            nn.BatchNorm2d(n_hidden),
            nn.LeakyReLU(slope, inplace=True)
        )
        
        self.downscale = nn.Sequential(
            DownscaleBlock(n_hidden, slope),
            DownscaleBlock(2*n_hidden, slope),
            DownscaleBlock(4*n_hidden, slope),
            conv3s(8*n_hidden, 8*n_hidden, 2),
            nn.BatchNorm2d(8*n_hidden),
            nn.LeakyReLU(slope, inplace=True)
        )
        
        self.fc = nn.Sequential(
            nn.AdaptiveMaxPool2d((4,4)),
            nn.Flatten(),
            nn.Linear(128*n_hidden, 32*n_hidden),
            nn.LeakyReLU(slope, inplace=True),
            nn.Linear(32*n_hidden, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        xi = self.proc_input(x)
        xd = self.downscale(xi)
        xc = self.fc(xd)
        return xc

if __name__ == "__main__":
    
    x = torch.randn(8, 3, 64, 64)
    
    slope = 0.1
    
    model = DiscriminatorModel(3, 32, slope)
    with torch.no_grad():
        y = model(x)
    print(y.shape)
