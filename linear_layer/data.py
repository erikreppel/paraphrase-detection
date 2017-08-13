# %%
import torch
import torch.utils.data as data
from nltk import tokenize
import pandas as pd
import numpy as np
from gensim.models.keyedvectors import KeyedVectors


def make_dataset(params):

    train = pd.read_csv('/home/erik/datasets/ppdb_train.csv')
    valid = pd.read_csv('/home/erik/datasets/ppdb_valid.csv')
    test = pd.read_csv('/home/erik/datasets/ppdb_test.csv')

    lookup = KeyedVectors.load_word2vec_format(params.word2vec_path,
                                               binary=True)
    vec_len = len(lookup['king'])
    train = PPDB(train, lookup, vec_len)
    valid = PPDB(valid, lookup, vec_len)
    test = PPDB(test, lookup, vec_len)
    return train, test, valid


# %%
def prep_sequence(seq, lookup, size):
    '''Converts a sequence to a vector'''
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(seq)
    sentence_vec = np.zeros((size,))
    for token in tokens:
        if token in lookup:
            sentence_vec += lookup[token]

    return torch.from_numpy(sentence_vec).float()


class MSRPSentVectorsDataSet(data.Dataset):
    def __init__(self, msrp_data, vocab, vec_length):
        self.X1, self.X2 = [], []

        for s1, s2 in zip(msrp_data['str1'], msrp_data['str2']):
            s1 = prep_sequence(s1, vocab, vec_length)
            s2 = prep_sequence(s2, vocab, vec_length)

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


class PPDB(data.Dataset):
    def __init__(self, ppdb_df, vocab, max_sentence_len):
        self.X1, self.X2 = [], []
        self.vocab = vocab

        for s1, s2 in zip(ppdb_df['str1'], ppdb_df['str2']):
            s1 = prep_sequence(s1, vocab, max_sentence_len)
            s2 = prep_sequence(s2, vocab, max_sentence_len)
            self.X1.append(s1)
            self.X2.append(s2)

        self.y = ppdb_df['paraphrase'].values

        assert(len(self.X1) == len(self.X2))
        assert(len(self.X1) == len(self.y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        xi1, xi2, yi = self.X1[i], self.X2[i], self.y[i]
        return xi1, xi2, yi