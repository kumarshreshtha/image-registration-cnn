"""module to keep track of loss improvement during training."""

class EarlyStopping(object):
    """Keeps state for the best loss value."""
    def __init__(self, patience:int):
        """

        Args:
            patience (int): Number of epochs to wait for improvement in loss 
                before stopping training.
        """
        self.patience = patience
        self._count = 0
        self.best_loss = None

    def __call__(self, val_loss:float)->bool:
        """Returns `True` if training should be stopped, `False` otherwise.
        
        Args:
            val_loss(float): the validation loss for the current training.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        if val_loss > self.best_loss:
            self._count += 1
            return self._count >= self.patience
        self.best_loss = val_loss
        self._count = 0
        return False