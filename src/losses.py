import torch
from torch import nn

class WMSE(nn.Module):

    def __init__(self, scaler=0.001):
        super(WMSE, self).__init__()
        self.scaler = scaler

    def weight(self, targets):
        t0 = (targets == 0).to(torch.int)
        weights = t0 * self.scaler + (1 - t0)
        return weights

    def forward(self, inputs, targets):
        weights = self.weight(targets)
        se = ((inputs - targets) ** 2) * weights
        return se.mean()