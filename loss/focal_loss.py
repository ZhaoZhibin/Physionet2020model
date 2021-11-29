import torch
import torch.nn as nn
import torch.nn.functional as F

class binary_focal_loss(nn.Module):

    def __init__(self, pos_weight, balance_param=1, gamma=2):
        super(binary_focal_loss, self).__init__()

        self.pos_weight = pos_weight
        self.balance_param = balance_param
        self.gamma = gamma

    def forward(self, input, target):
        assert target.size() == input.size()

        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(input, target, pos_weight=self.pos_weight, reduction='none')
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        balanced_focal_loss = (self.balance_param * focal_loss).mean()
        return balanced_focal_loss
