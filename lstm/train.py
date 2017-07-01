from time import time
import os

import pandas as pd
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from preprocess import prep_sentence_couples
from model import SiameseRNN


# Variables ###################################################################
checkpoint_path = '/home/erik/Desktop/sia_checkpoints'
train_path = '../small_data/train.csv'
test_path = '../small_data/test.csv'
lr = 20
batch_size = 32
USE_GPU = True
start = str(time())

# Pre-process data ############################################################
print('Pre-processing data')
word_vectors = KeyedVectors.load_word2vec_format(
    '~/datasets/GoogleNews-vectors-negative300.bin',
    binary=True)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

X_train1, X_train2 = prep_sentence_couples(train['str1'],
                                           train['str2'],
                                           word_vectors)
X_test1, X_test2 = prep_sentence_couples(test['str1'],
                                         test['str2'],
                                         word_vectors)

y_train = train['paraphrase'].values
y_test = test['paraphrase'].values

# pytorch-ify the vectors######################################################
X_train1 = torch.from_numpy(X_train1).float()
X_train2 = torch.from_numpy(X_train2).float()

X_test1 = torch.from_numpy(X_test1).float()
X_test2 = torch.from_numpy(X_test2).float()

y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()

if USE_GPU:
    X_train1 = X_train1.cuda()
    X_train2 = X_train2.cuda()
    y_train = y_train.cuda()
    X_test1 = X_test1.cuda()
    X_test2 = X_test2.cuda()
    y_test = y_test.cuda()


# Create model ################################################################
print('Creating model')
model = SiameseRNN(input_size=300,
                   hidden_size=200,
                   num_layers=3,
                   batch_size=batch_size,
                   gpu=USE_GPU)
if USE_GPU:
    print('Using GPU')
    model.cuda()

criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters())


# Train the model #############################################################
def get_batch(X1, X2, y, i, evaluation=False):
    global batch_size
    local_batch_size = min(batch_size, len(y) - 1 - i)
    xi1 = Variable(X1[i:i+local_batch_size], volatile=evaluation)
    xi2 = Variable(X2[i:i+local_batch_size], volatile=evaluation)
    yi = Variable(y[i:i+local_batch_size])
    return xi1, xi2, yi


def train(X1, X2, y):
    model.train()
    total_loss = 0
    for batch, i in enumerate(range(0, y.size(0) - 1, batch_size)):
        h1, h2 = model.init_hidden()
        if (len(X1) - i) < batch_size:
            continue
        x1i, x2i, targets = get_batch(X1, X2, y, i)

        output, h1, h2 = model(x1i, x2i, h1, h2)

        optimizer.zero_grad()
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.data[0]

    return total_loss


def calc_accuracy(y, y_hat):
    y, y_hat = y.data.clone().cpu().numpy(), y_hat.data.clone().cpu().numpy()
    dif = y - y_hat
    wrong = 0.0
    for d in dif:
        if d == 0:
            wrong += 1
    acc = float(wrong) / len(y_hat)
    std = np.std(y_hat)

    return 1 - acc, std


def evaluate(X1, X2, y):
    model.eval()
    total_loss = 0.0
    batches_ran = 0.0
    total_acc = 0.0
    for batch, i in enumerate(range(0, y.size(0) - 1, batch_size)):
        h1, h2 = model.init_hidden()
        if (len(X1) - i) < batch_size:
            continue
        x1i, x2i, targets = get_batch(X1, X2, y, i, evaluation=True)

        output, h1, h2 = model(x1i, x2i, h1, h2)

        total_loss += len(y) * criterion(output, targets).data
        acc, std = calc_accuracy(targets, output)
        total_acc += acc
        batches_ran += 1

    avg_loss = total_loss / batches_ran
    return {
        'total_loss': total_loss[0],
        'avg_loss': avg_loss[0],
        'avg_acc': total_acc / batches_ran,
        'std_of_answer': std
    }


def checkpoint(epoch, model):
    with open(
        os.path.join(checkpoint_path,
                     'siamese_lstm_epoch-{}-{}.pt'.format(start,
                                                          epoch)), 'wb') as f:
        torch.save(model, f)


best_val_loss = None

print('Starting to train')

try:
    for epoch in range(1, 10000):
        start_time = time()

        train_loss = train(X_train1, X_train2, y_train)
        print("Epoch {} -- train loss = {}".format(epoch, train_loss))

        eval_results = evaluate(X_test1, X_test2, y_test)
        print('-- Eval stats:\n \t', eval_results)

        time_taken = time() - float(start_time)
        print('epoch took {}s'.format(time_taken))
        print("-"*100)

        if not best_val_loss or best_val_loss > eval_results['total_loss']:
            best_val_loss = eval_results['total_loss']
        else:
            lr /= 4.0

        if epoch % 50 == 0:
            checkpoint(epoch, model)


except KeyboardInterrupt:
    print('Keyboard interupt received, saving model and finishing')
    checkpoint('exit', model)
