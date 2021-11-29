import torch
from torch import nn


class DiceLoss(nn.Module):
    """DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
    Useful in dealing with unbalanced data
    Add softmax automatically
    """

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.m = nn.Sigmoid()
        self.gamma = 1.
        self.p = 2

    def forward(self, y_pred, y_true):
        # shape(y_pred) = batch_size, label_num, **
        # shape(y_true) = batch_size, **
        # y_pred = torch.softmax(y_pred, dim=1)
        pred_prob = self.m(y_pred)
        # pred_prob = torch.gather(y_pred, dim=1, index=y_true.unsqueeze(1))
        numerator = 2. * torch.sum(pred_prob * y_true, dim=1) + self.gamma
        denominator = torch.sum(pred_prob.pow(self.p) + y_true, dim=1) + self.gamma
        dsc_i = 1. - numerator / denominator
        dice_loss = dsc_i.mean()
        return dice_loss

class TverskyLoss(nn.Module):
    """DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
    Useful in dealing with unbalanced data
    Add softmax automatically
    """

    def __init__(self):
        super(TverskyLoss, self).__init__()
        self.m = nn.Sigmoid()
        self.gamma = 1.
        self.p = 2
        self.alpha = 0.7

    def forward(self, y_pred, y_true):
        # shape(y_pred) = batch_size, label_num, **
        # shape(y_true) = batch_size, **
        # y_pred = torch.softmax(y_pred, dim=1)
        pred_prob = self.m(y_pred)
        # pred_prob = torch.gather(y_pred, dim=1, index=y_true.unsqueeze(1))
        true_pos = torch.sum(pred_prob * y_true, dim=1)
        #false_neg = torch.sum((1-pred_prob) * y_true, dim=1)
        #false_pos = torch.sum(pred_prob * (1-y_true), dim=1)
        numerator = true_pos + self.gamma
        #denominator = true_pos + self.alpha*false_neg + (1-self.alpha)*false_pos + self.gamma

        denominator = torch.sum((1-self.alpha)*pred_prob.pow(self.p) + self.alpha*y_true, dim=1) + self.gamma
        tl_i = (1. - numerator / denominator).pow(0.75)
        tl_loss = tl_i.mean()
        return tl_loss