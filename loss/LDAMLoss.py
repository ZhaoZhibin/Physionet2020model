import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        self.max_m = max_m
        self.cls_num_list = cls_num_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, pred, label):
        loss = 0
        for i in range(pred.shape[1]):
            #x = torch.stack((pred[:, i], torch.zeros_like(pred[:, i])), 1)
            x = pred[:, i]
            target = label[:, i]
            index = torch.zeros_like(pred[:, 0:2], dtype=torch.uint8)
            #index = torch.zeros((x.shape[0], 2), dtype=torch.uint8)
            index.scatter_(1, target.data.view(-1, 1).type(torch.cuda.LongTensor), 1)

            index_float = index.type(torch.cuda.FloatTensor)

            cls_num_list = np.array(self.cls_num_list[i])
            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (self.max_m / np.max(m_list))
            m_list[0] = m_list[0]*0
            m_list = torch.cuda.FloatTensor(m_list)

            batch_m = torch.matmul(m_list[None, :], index_float.transpose(0, 1))
            batch_m = batch_m.view((-1, ))
            #print(x.shape, batch_m.shape)
            x_m = x - batch_m

            #output = torch.where(index, x_m, x)
            weight = self.weight[i]
            loss = loss + F.binary_cross_entropy_with_logits(self.s * x_m, target, pos_weight=weight)
        loss = loss / pred.shape[1]
        return loss
