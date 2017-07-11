from torch.autograd import Variable
import torch.nn as nn
import torch
GPU = torch.cuda.is_available()


# %%
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class SiaLSTM(nn.Module):
    '''Siamese LSTM with word embedding'''
    def __init__(self, embedding_dim, hidden_dim, layer_dim,
                 output_dim, vocab_size):
        super(SiaLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.embedder = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim,
                             layer_dim, batch_first=True)

        self.lstm2 = nn.LSTM(embedding_dim, hidden_dim,
                             layer_dim, batch_first=True)

    def forward(self, x1, x2):
        '''defines the forward pass of the neural network'''
        h1, c1, h2, c2 = self.init_hidden(x1.size(0))
        x1 = self.embedder(x1)
        x2 = self.embedder(x2)

        x1, (h1n, c1n) = self.lstm1(x1, (h1, c1))
        x2, (h2n, c2n) = self.lstm2(x2, (h2, c2))

        x1, x2 = x1[:, -1, :], x2[:, -1, :]

        out = cosine_similarity(x1, x2)
        return out

    def init_hidden(self, size):
        '''Init the hidden states of the lstm'''
        if GPU:
            h1 = Variable(torch.zeros(self.layer_dim, size, self.hidden_dim).cuda())
            c1 = Variable(torch.zeros(self.layer_dim, size, self.hidden_dim).cuda())
            h2 = Variable(torch.zeros(self.layer_dim, size, self.hidden_dim).cuda())
            c2 = Variable(torch.zeros(self.layer_dim, size, self.hidden_dim).cuda())
        else:
            h1 = Variable(torch.zeros(self.layer_dim, size, self.hidden_dim))
            c1 = Variable(torch.zeros(self.layer_dim, size, self.hidden_dim))
            h2 = Variable(torch.zeros(self.layer_dim, size, self.hidden_dim))
            c2 = Variable(torch.zeros(self.layer_dim, size, self.hidden_dim))
        return h1, c1, h2, c2


# %%
X1 = torch.rand(32, 25)
X2 = torch.rand(32, 25)
print(X1.size(), X2.size())
cosine_similarity(X1, X2)