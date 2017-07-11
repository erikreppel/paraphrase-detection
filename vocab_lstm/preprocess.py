# %%
# from sklearn.preprocessing import OneHotEncoder
from nltk import tokenize
import torch
import torch.utils.data as data


# %%
def build_vocab(train, test):
    vocab = []
    vocab += _tokenize(train['str1'])
    vocab += _tokenize(train['str2'])
    vocab += _tokenize(test['str1'])
    vocab += _tokenize(test['str2'])
    lookup = {token: i+1 for i, token in enumerate(set(vocab))}
    lookup[""] = 0
    rev_lookup = {i+1: token for i, token in enumerate(set(vocab))}
    rev_lookup[0] = ""
    return lookup, rev_lookup


def _tokenize(sentences):
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = []
    for sentence in sentences:
        tokens += tokenizer.tokenize(sentence)
    return tokens


# %%
def prep_sequence(seq, lookup, max_len, gpu=False):
    '''Converts a sequence to a vector'''
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(seq)
    seq = [lookup[s] for s in tokens]
    while len(seq) < max_len:
        seq.append(0)
    if gpu:
        return torch.LongTensor(seq).cuda()
    else:
        return torch.LongTensor(seq)


def longest_sentence_length(data1, data2):
    max_length = 0
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    for sent_list in [data1['str1'], data1['str2'],
                      data2['str1'], data2['str2']]:
        for sent in sent_list:
            tokens = tokenizer.tokenize(sent)
            if len(tokens) > max_length:
                max_length = len(tokens)

    return max_length


# %%
class MSRPWordVectorsDataSet(data.Dataset):
    def __init__(self, msrp_data, vocab, max_sentence_len, gpu=False):
        self.gpu = gpu

        self.X1, self.X2 = [], []

        for s1, s2 in zip(msrp_data['str1'], msrp_data['str2']):
            s1 = prep_sequence(s1, vocab, max_sentence_len, gpu)
            s2 = prep_sequence(s2, vocab, max_sentence_len, gpu)

            self.X1.append(s1)
            self.X2.append(s2)

        self.y = msrp_data['paraphrase'].values

        assert(len(self.X1) == len(self.X2))
        assert(len(self.X1) == len(self.y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        xi1, xi2, yi = self.X1[i], self.X2[i], self.y[i]
        return xi1, xi2, yi
