import torch
from torch import nn


class EntropyLoss(nn.Module):

    # implementaiton of simple entropy loss for Goyal

    def __init__(self):
        super().__init__()

    def forward(self, a, m):
        # a = (N,)
        # m = (N, N) , in our case m is P (like in article)
        # however, by applying entropy losses on a and rows of P
        # (minimizing their entropy), we can recover a nearly binary solution for a and the rows of P

        a_ = torch.softmax(a, dim=-1) * torch.log_softmax(a, dim=-1)
        a_ = -1.0 * a_.sum()

        m_ = torch.softmax(m, dim=1) * torch.log_softmax(m, dim=1)
        m_ = -1.0 * m_.sum()
        return a_ + m_