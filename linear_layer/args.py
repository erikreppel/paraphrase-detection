import argparse
import time
import torch

start = time.strftime("%m-%d-%H:%M")
GPU = torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='./logs/runs/' + start, help='location of tensorboard logs')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/' + start, help='location of model checkpoints')
parser.add_argument('--model_path', type=str, default='', help='path of model existing model to base off of')
parser.add_argument('--train_path', type=str, default='./data/train.csv', help='path to train data')
parser.add_argument('--ppdb_path', type=str, default='/home/erik/datasets/parsed_ppdb.csv', help='path to ppdb')
parser.add_argument('--test_path', type=str, default='./data/test.csv', help='path to test data')
parser.add_argument('--word2vec_path', type=str, default='~/datasets/GoogleNews-vectors-negative300.bin', help='path to pretrained vectors for word2vec')

parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--n_epochs', type=int, default=10000, help='number of epochs')


parser.add_argument('--cuda', type=bool, default=GPU, help='enables gpu usage (defaults to true if gpu detected)')
parser.add_argument('--n_workers', type=int, default=2, help='number of worker threads to use for loading data')

parser.add_argument('--seed', type=int, default=42, help='number to seed randomness')

parser.add_argument('--train_eval_freq', type=int, default=100, help='How frequently to evaluate training performance')
parser.add_argument('--valid_eval_freq', type=int, default=100, help='How frequently to evaluate performance on validation set (typically every epoch)')
parser.add_argument('--checkpoint_freq', type=int, default=1000, help='How frequently to save a checkpoint of the model')