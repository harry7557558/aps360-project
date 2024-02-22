import torch
import torch.nn as nn

def conv3s(nin, nout, stride):
    return nn.Conv2d(nin, nout, 3, stride=stride, padding=1)

def lrelu(x, slope, inplace=True):
    return nn.functional.leaky_relu(x, slope, inplace=inplace)
