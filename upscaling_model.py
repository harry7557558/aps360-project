import torch
import torch.nn as nn


def conv3(nin, nout):
    return nn.Conv2d(nin, nout, 3, padding=1)

def relu(x, inplace=True):
    return nn.functional.relu(x, inplace=inplace)


class ResidualBlock(nn.Module):
    def __init__(self, n):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3(n, n)
        self.conv2 = conv3(n, n)

    def forward(self, x):
        return self.conv2(relu(self.conv1(x))) + x


class UpscalingModel(nn.Module):

    def __init__(self, scale, n_in, n_out, n_hidden, n_residual):
        super(UpscalingModel, self).__init__()

        # input
        self.conv_i = conv3(n_in, n_hidden)

        # residual
        self.residual = nn.Sequential(
            *[ResidualBlock(n_hidden) for _ in range(n_residual)]
        )
        self.conv_m = conv3(n_hidden, n_hidden)

        # upscale
        if scale == 1:
            self.upscale = lambda _: _
        elif scale == 2:
            self.upscale = nn.Sequential(
                conv3(n_hidden, 4*n_hidden),
                nn.ReLU(),
                nn.PixelShuffle(2)
            )
        elif scale == 3:
            self.upscale = nn.Sequential(
                conv3(n_hidden, 9*n_hidden),
                nn.ReLU(),
                nn.PixelShuffle(3)
            )
        elif scale == 4:
            self.upscale = nn.Sequential(
                conv3(n_hidden, 4*n_hidden),
                nn.ReLU(),
                nn.PixelShuffle(2),
                conv3(n_hidden, 4*n_hidden),
                nn.ReLU(),
                nn.PixelShuffle(2)
            )
        else:
            raise ValueError("Scale must be 2, 3, or 4")

        # output
        # self.conv_o = nn.Conv2d(n_hidden, n_out, 7, padding=3)
        self.conv_o = nn.Conv2d(n_hidden, n_out, 3, padding=1)

    def forward(self, x):
        # input
        x_i = relu(self.conv_i(x))
        # residual
        x_r = self.residual(x_i)
        x_m = self.conv_m(x_r) + x_i
        # upscale
        x_u = self.upscale(x_m)
        # output
        x_o = torch.sigmoid(self.conv_o(x_u))
        return x_o


if __name__ == "__main__":

    # make sure there is no error

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.randn((8, 3, 64, 64), device=device)

    model = UpscalingModel(1, 3, 3, 32, 10).to(device)
    with torch.no_grad():
        y = model(x)
    print(y.shape)

    model = UpscalingModel(2, 3, 3, 32, 10).to(device)
    with torch.no_grad():
        y = model(x)
    print(y.shape)

    model = UpscalingModel(3, 3, 3, 32, 10).to(device)
    with torch.no_grad():
        y = model(x)
    print(y.shape)

    model = UpscalingModel(4, 3, 3, 32, 10).to(device)
    with torch.no_grad():
        y = model(x)
    print(y.shape)
