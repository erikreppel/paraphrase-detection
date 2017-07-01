import torch
import torch.nn as nn
from torch.autograd import Variable


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class SiameseRNN(nn.Module):
    def __init__(self,
                 input_size=300,
                 hidden_size=200,
                 num_layers=3,
                 output_size=1,
                 batch_size=10,
                 gpu=False):
        super(SiameseRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = batch_size
        self.batch_size = batch_size
        self.gpu = gpu

        self.rnn1 = nn.RNN(input_size=self.input_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           batch_first=True)

        self.rnn2 = nn.RNN(input_size=self.input_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           batch_first=True)

        # self.linear_layer1 = nn.Linear(self.hidden_size, self.output_size)
        # self.linear_layer2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x1, x2, h1, h2):
        out1, h1 = self.rnn1(x1, h2)
        out2, h2 = self.rnn1(x1, h2)

        in_linear1 = out1[:, -1, :]
        in_linear2 = out2[:, -1, :]

        dist = cosine_similarity(in_linear1, in_linear2)
        return dist, h1, h2

    def init_hidden(self):
        h0 = Variable(torch.zeros((self.num_layers,
                                   self.batch_size,
                                   self.hidden_size)))
        h1 = Variable(torch.zeros((self.num_layers,
                                   self.batch_size,
                                   self.hidden_size)))
        if self.gpu:
            h0, h1 = h0.cuda(), h1.cuda()
        return h0, h1