import torch
from tensorboard_logger import log_value


def checkpoint(model, checkpoint_path):
    '''Save model checkpoint'''
    with open(checkpoint_path, 'wb') as f:
        torch.save(model.state_dict(), f)


# tensorboard_log_test, tensorboard_log_valid, and tensorboard_log_train should
# be changed to log metrics you actually care about.
def tensorboard_log_test(accuracy, loss, std):
    '''Wraps logging of test metrics'''
    log_value('test_loss', loss, 1)
    log_value('test_accuracy', accuracy, 1)
    log_value('test_pred_std', std, 1)


def tensorboard_log_valid(epoch, accuracy, loss, std):
    '''Wraps logging of validation metrics'''
    log_value('valid_loss', loss, epoch)
    log_value('valid_accuracy', accuracy, epoch)
    log_value('valid_pred_std', std, epoch)


def tensorboard_log_train(epoch, loss, acc):
    '''Wraps logging of training metrics'''
    log_value('train_loss', loss, epoch)
    log_value('train_accuracy', acc, epoch)


def r2_accuracy(pred, target):
    '''Computes R2 accuracy between target values and predictions'''
    y_mean = target.mean()
    msse = torch.pow(target-y_mean, 2).sum()
    psse = torch.pow(target-pred, 2).sum()
    return 1 - (psse / msse)


def eval_train_performance(i, loss, pred, target, silent=False):
    '''
    Evaluates training performance (loss and R2 accuracy), logs to tensorboard
    params:
        - i: numeric counter associated with values (usually epoch)
        - loss: output Variable from a criterion
        - pred: prediction made by model
        - target: actual values
        - silent (default: False) whether or not to log to stdout
    '''
    accuracy = r2_accuracy(pred, target)

    tensorboard_log_train(i, loss, accuracy)
    if not silent:
        print('='*80)
        print('TRAIN: Epoch {} loss: {:.4f} accuracy: {:%}'.format(
            i, loss, accuracy
        ))
