# %%
import os
from time import time
import preprocess
from model import SiaLSTM

import pandas as pd

from torch.autograd import Variable
import torch.nn as nn
import torch

from tensorboard_logger import configure, log_value


# %%
# Variables ===================================================================
GPU = torch.cuda.is_available()
start = time()

batch_size = 32
n_epochs = 10000
embedding_dim = 500
hidden_dim = 126
layer_dim = 2
output_dim = 1
seq_dim = embedding_dim

tensor_board_log_dir = '/home/erik/Desktop/sia_logs/runs/' + str(start) + '/'
checkpoint_path = '/home/erik/Desktop/sia_checkpoints'
train_path = '../small_data/train.csv'
test_path = '../small_data/test.csv'

configure(tensor_board_log_dir)


# %%
# Pre-process the data ========================================================
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

word_2_int, int_2_word = preprocess.build_vocab(train, test)
max_sent_len = preprocess.longest_sentence_length(train, test)

print('Longest sentence: {}'.format(max_sent_len))

train = preprocess.MSRPWordVectorsDataSet(train, word_2_int, max_sent_len, GPU)
test = preprocess.MSRPWordVectorsDataSet(test, word_2_int, max_sent_len, GPU)


train_loader = torch.utils.data.DataLoader(dataset=train,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test,
                                          batch_size=batch_size,
                                          shuffle=False)

# %%
# Create model, criterion, optimizer ==========================================
model = SiaLSTM(embedding_dim, hidden_dim, layer_dim,
                output_dim, len(word_2_int))
if GPU:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.L1Loss()


# Utils =======================================================================
def checkpoint(epoch, model):
    with open(
        os.path.join(checkpoint_path,
                     'siamese_lstm_epoch-{}-{}.pt'.format(start,
                                                          epoch)), 'wb') as f:
        torch.save(model, f)


def tensor_log(epoch, accuracy, loss, std):
    log_value('valid_loss', loss, epoch)
    log_value('valid_accuracy', accuracy, epoch)
    log_value('valid_pred_std', std, epoch)


def tensor_log_train(epoch, loss):
    log_value('train_loss', std, epoch)


# %%
# Train =======================================================================
for epoch in range(1, n_epochs):
    for i, (X1, X2, labels) in enumerate(train_loader):

        X1 = Variable(X1)
        X2 = Variable(X2)
        labels = Variable(labels).float().cuda()

        optimizer.zero_grad()
        outputs = model(X1, X2)

        loss = criterion(outputs, labels)
        tensor_log_train(i*epoch, loss.data[0])

        loss.backward()
        optimizer.step()

    if epoch % 5 == 0:
        model.eval()
        correct, total = 0, 0
        for X1, X2, labels in test_loader:

            X1 = Variable(X1)
            X2 = Variable(X2)
            labels = Variable(labels).float().cuda()

            outputs = model(X1, X2)
            std = outputs.std().data[0]
            predicted = outputs.round()
            total += labels.size(0)

            if GPU:
                same = predicted.cpu() == labels.cpu()
            else:
                same = predicted == labels
            c = same.int().sum()
            correct += c[0][0]

        accuracy = float(correct.data[0]) / total
        loss = loss.data[0]
        print('=' * 80)
        print('Epoch {} loss: {} accuracy: {:%} std: {}'.format(
            epoch,
            loss,
            accuracy,
            std))
        model.train()
        tensor_log(epoch, accuracy, loss, std)

    if epoch % 10 == 0:
        checkpoint(epoch, model)
