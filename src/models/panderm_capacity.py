"""Compact nonlinear readouts, including a preserved linear baseline path."""
import numpy as np
import torch
from torch import nn


class ResidualHead(nn.Module):
    def __init__(self,coef,intercept,widths,scale):
        super().__init__()
        self.linear=nn.Linear(1024,7)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor(np.asarray(coef),dtype=torch.float32))
            self.linear.bias.copy_(torch.tensor(np.asarray(intercept),dtype=torch.float32))
        self.linear.requires_grad_(False)
        layers=[];previous=1024
        for width in widths:
            layers.extend([nn.Linear(previous,width),nn.GELU(),nn.Dropout(.2)])
            previous=width
        layers.append(nn.Linear(previous,7))
        self.correction=nn.Sequential(*layers)
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.scale=scale

    def forward(self,x):
        return self.linear(x)+self.scale*self.correction(x)


def plain_head(width):
    return nn.Sequential(nn.Linear(1024,width),nn.GELU(),nn.Dropout(.2),nn.Linear(width,7))
