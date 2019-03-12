# __init__.py
from .base import Callback, Callbacks
from .history import History
from .early_stopping import EarlyStopping



""" SKETCH OF TRAINING LOOP
"""
import torch
import pytorch_nns.helpers as h
import pytorch_nns.metrics as metrics
INPT_KEY='input'
TARG_KEY='target'

BASE_STATE_ATTRIBUTES=[
    'nb_epochs',
    'mode',
    'epoch',
    'batch' ]

COMPUTED_STATE_ATTRIBUTES=[
    'batch_loss',
    'batch_acc', 
    'total_loss',
    'loss',
    'acc',
    'val_batch_loss',
    'val_batch_acc', 
    'val_total_loss',
    'val_loss',
    'val_acc' ]

STATE_ATTRIBUTES=BASE_STATE_ATTRIBUTES+COMPUTED_STATE_ATTRIBUTES
CALLBACK_ERROR='Trainer: callbacks already set. use force=True to override'




def batch_accuracy(round_prediction):
    def _calc(outputs,targets):
        return metrics.accuracy(
            outputs,
            targets,
            argmax=(not round_prediction),
            round_prediction=round_prediction,
            axis=1)
    return _calc





class Trainer(object):

    def __init__(self,
            model,
            criterion=None,
            optimizer=None,
            force_cpu=False):
        self.device=h.get_device(force_cpu)
        self.model=model.to(self.device)
        self.callbacks=False
        if criterion and optimizer:
            self.compile(criterion=criterion,optimizer=optimizer)


    def compile(self,criterion,optimizer):
        self.criterion=criterion
        self.optimizer=optimizer


    def set_callbacks(self,
            callbacks=[],
            force=False,
            history_callback=True,
            **history_kwargs):
        if force or (self.callbacks is False):
            if isinstance(callbacks,list):
                callbacks=Callbacks(callbacks)
            if history_callback:
                callbacks.append(History(**history_kwargs))
            self.callbacks=callbacks
        else:
            raise ValueError(CALLBACK_ERROR)


    def fit(self,
            train_loader,
            valid_loader=None,
            nb_epochs=1,
            accuracy_method=None,
            round_prediction=False,
            initial_loss=9e12):
        self._reset_state(nb_epochs=nb_epochs)
        self.best_loss=initial_loss
        self.best_epoch=0
        self.accuracy_method=accuracy_method or batch_accuracy(round_prediction)
        self.callbacks.on_train_begin(**self._state())
        for epoch in range(1,nb_epochs+1):
            self._reset_state(epoch=epoch)
            self._run_epoch(
                epoch=epoch,
                loader=train_loader,
                mode='train')
            # validation
            if valid_loader:
                with torch.no_grad():
                    self.callbacks.on_validation_begin(**self._state())
                    self._run_epoch(
                        epoch=epoch,
                        loader=valid_loader,
                        mode='valid')
                    self.callbacks.on_validation_end(**self._state())
                self._check_for_best(epoch=epoch,loss=self.val_loss)
            else:
                self._check_for_best(epoch=epoch,loss=self.loss)
            self.callbacks.on_epochs_complete(**self._state())
        self.callbacks.on_train_end(**self._state())


    def _reset_state(self,**kwargs):
        for attr in STATE_ATTRIBUTES:
            setattr(self,attr,kwargs.get(attr,0))


    def _state(self):
        state={}
        for attr in STATE_ATTRIBUTES:
            state[attr]=getattr(self,attr)       
        return state


    def _update_state(self,**kwargs):
        for attr in BASE_STATE_ATTRIBUTES:
            val = kwargs.get(attr,None)
            if val: setattr(self,attr,val)
        if kwargs.get('batch_loss') and kwargs.get('batch_acc'):
            self._set_scores(**kwargs)


    def _set_scores(self,mode=None,batch_loss=None,batch_acc=None,**kwargs):
        if mode=='train':
            self.batch_loss=batch_loss
            self.batch_acc=batch_acc
            self.total_loss+=self.batch_loss
            self.loss=self.total_loss/self.batch
            self.acc=((self.acc*(self.batch-1))+self.batch_acc)/self.batch
        else:
            self.val_batch_loss=batch_loss
            self.val_batch_acc=batch_acc
            self.val_total_loss+=self.val_batch_loss
            self.val_loss=self.val_total_loss/self.batch
            self.val_acc=((self.val_acc*(self.batch-1))+self.val_batch_acc)/self.batch


    def _run_epoch(self,
            epoch,
            loader,
            mode):
        self._update_state(mode=mode)
        self.callbacks.on_epoch_begin(**self._state())
        for i,batch in enumerate(loader):
            self._run_batch(i+1,batch,mode)
        self.callbacks.on_epoch_end(**self._state())


    def _run_batch(self,batch_index,batch,mode):
        self._update_state(batch=batch_index)
        self.callbacks.on_batch_begin()
        inputs, targets=self._batch_data(batch)
        self.callbacks.on_forward_begin(**self._state())
        outputs=self.model(inputs)
        self.callbacks.on_forward_end(**self._state())
        loss=self.criterion(outputs,targets)
        self._update_state(
            mode=mode,
            batch_loss=loss.item(),
            batch_acc=self.accuracy_method(outputs,targets).item())
        self.callbacks.on_loss_computed(**self._state())
        if mode=='train':
            self.optimizer.zero_grad()
            self.callbacks.on_backward_begin(**self._state())
            loss.backward()
            self.callbacks.on_backward_end(**self._state())
            self.optimizer.step()
        self.callbacks.on_batch_end(**self._state())

            
    def _batch_data(self,batch):
        inputs=batch[INPT_KEY].float().to(self.device)
        targets=batch[TARG_KEY].float().to(self.device)
        return inputs, targets


    def _check_for_best(self,epoch,loss):
        if self.best_loss>loss:
            self.best_loss=loss
            self.best_epoch=epoch








