import torch.nn as nn
import torch


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class LinearDifference(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearDifference, self).__init__()

        hidden_dim = int(output_dim/2)

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)

        self.l2 = nn.Linear(input_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x1, x2):
        x1 = self.l1(x1)
        x1 = self.l3(x1)

        x2 = self.l2(x2)
        x2 = self.l4(x2)

        x = cosine_similarity(x1, x2)
        return x
