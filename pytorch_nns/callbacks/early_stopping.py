from .base import Callback


class EarlyStopping(Callback):


    def __init__(self,
            patience=0,
            epoch_start=0):
        super(EarlyStopping,self).__init__()
        self.patience=patience
        self.epoch_start=epoch_start
        self.best_loss=0
        self.best_epoch=0
        self.patience_count=0


    def on_epochs_complete(self,**kwargs):
        if kwargs['epoch']>=self.epoch_start:
            if self.best_loss>=kwargs['best_loss']:
                self.best_loss=kwargs['best_loss']
                self.best_epoch=kwargs['epoch']
                self.patience_count=0
            else:
                self.patience_count+=1
        if self.patience_count>self.patience:
            break
