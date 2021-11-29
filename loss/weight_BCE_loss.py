
from torch import nn
import torch

class WeightedMultilabel(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(pos_weight=weights)

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        return loss
