# %%
import os
from tensorboard_logger import configure

from args import parser
import models
import data
import utils

from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as dd
import torch

# Setup arguments and constants ===============================================
# %%
params = parser.parse_args()
try:
    os.mkdir(params.checkpoint_dir)
    os.mkdir(params.log_dir)
except FileExistsError:
    pass

configure(params.log_dir)

input_dim = 300
output_dim = 512

print('Starting with params:', params)


# Pre-process data and create dataloaders =====================================
# %%
print('Pre-processing data')
(train, valid, test) = data.make_dataset(params)

train_loader = dd.DataLoader(dataset=train, batch_size=params.batch_size,
                             shuffle=True)
valid_loader = dd.DataLoader(dataset=valid, batch_size=params.batch_size,
                             shuffle=True)
test_loader = dd.DataLoader(dataset=test, batch_size=params.batch_size,
                            shuffle=True)


# Create model, optimizer, and criterion ======================================
print('Creating model')
model = models.LinearDifference(input_dim, output_dim)
if params.model_path != '':
    model.load_state_dict(torch.load(params.model_path))

if params.cuda:
    model.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

print('Set optimizer as {} and loss function as {}'.format(
    type(optimizer).__name__,
    type(criterion).__name__)
)


# Define train, validate, and test ============================================
# %%
def train(epoch):
    model.train()
    for i, (X1, X2, y) in enumerate(train_loader):
        X1 = Variable(X1)
        X2 = Variable(X2)
        y = Variable(y).float()
        if params.cuda:
            X1 = X1.cuda()
            X2 = X2.cuda()
            y = y.cuda()

        optimizer.zero_grad()
        y_hat = model(X1, X2)
        loss = criterion(y_hat, y)

        loss.backward()
        optimizer.step()

    if epoch % params.train_eval_freq == 0:
        utils.eval_train_performance(epoch, loss.data[0], y_hat.data, y.data)


def valid(epoch):
    model.eval()
    n_batches = 0.
    total_accuracy = 0
    total_loss = 0
    for i, (X1, X2, y) in enumerate(valid_loader):
        X1 = Variable(X1)
        X2 = Variable(X2)
        y = Variable(y).float()
        if params.cuda:
            X1 = X1.cuda()
            X2 = X2.cuda()
            y = y.cuda()

        y_hat = model(X1, X2)
        loss = criterion(y_hat, y)

        total_accuracy += utils.r2_accuracy(y_hat.data, y.data.view(-1, 1))
        total_loss += loss.data[0]
        n_batches += 1

    std = y_hat.std().data[0]
    avg_accuracy = total_accuracy / n_batches
    avg_loss = total_loss / n_batches

    utils.tensorboard_log_valid(epoch, avg_accuracy, avg_loss, std)
    print('*'*80)
    print('VALIDATE: Epoch {} loss: {:.4f} accuracy: {:%} std: {:.4f}'.format(
        epoch, avg_loss, avg_accuracy, std
    ))


def test():
    model.eval()
    n_batches = 0.
    total_accuracy = 0
    total_loss = 0
    for i, (X1, X2, y) in enumerate(test_loader):
        X1 = Variable(X1)
        X2 = Variable(X2)
        y = Variable(y).float()
        if params.cuda:
            X1 = X1.cuda()
            X2 = X2.cuda()
            y = y.cuda()

        y_hat = model(X1, X2)
        loss = criterion(y_hat, y)

        total_accuracy += utils.r2_accuracy(y_hat.data, y.data)
        total_loss += loss.data[0]
        n_batches += 1

    std = y_hat.std().data[0]
    avg_accuracy = total_accuracy / n_batches
    avg_loss = total_loss / n_batches

    utils.tensorboard_log_test(avg_accuracy, avg_loss, std)
    print('$'*80)
    print('TEST: Epoch {} loss: {:.4f} accuracy: {:%} std: {:.4f}'.format(
        epoch, avg_loss, avg_accuracy, std
    ))


# Run the experiment ==========================================================
try:
    for epoch in range(params.n_epochs):
        train(epoch)
        if epoch % params.valid_eval_freq == 0:
            valid(epoch)

        if epoch % params.checkpoint_freq == 0:
            checkpoint_path = os.path.join(params.checkpoint_dir,
                                           'model_' + str(epoch) + '.pt')
            utils.checkpoint(model, checkpoint_path)

    print('Training finished')
    checkpoint_path = os.path.join(params.checkpoint_dir, 'model_last.pt')
    utils.checkpoint(model, checkpoint_path)
    print('Model saved. Evaluating test set')
    test()

except KeyboardInterrupt:
    print('Keyboard Interrupt received. Saving, testing, and shutting down')
    checkpoint_path = os.path.join(params.checkpoint_dir, 'model_last.pt')
    utils.checkpoint(model, checkpoint_path)
    print('Model saved. Evaluating test set')
    test()
