import torch
import numpy as np


class EarlyStopping(object):
    def __init__(self, patience, verbose=False):
        self.patience = patience
        self.count = 0
        self.best_loss = None
        self.verbose = verbose
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, model, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch)
        elif val_loss > self.best_loss:
            self.count += 1
            if self.verbose:
                print(f'EarlyStopping counter:\
                     {self.count} out of {self.patience}')
            if self.count >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch)
            self.count = 0

    def save_checkpoint(self, val_loss, model, epoch):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f}\
                --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(),
                   f'./../checkpoints/checkpoint_{epoch}.pth')
        self.val_loss_min = val_loss
