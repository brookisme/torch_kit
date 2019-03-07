# __init__.py
from .base import Callback, Callbacks
from .core import History



""" SKETCH OF TRAINING LOOP
"""
import torch
import pytorch_nns.helpers as h
import pytorch_nns.metrics as metrics
INPT_KEY='input'
TARG_KEY='target'

BASE_STATE_ATTRIBUTES=[
    'nb_epochs',
    'epoch',
    'batch',
    'batch_loss',
    'batch_acc' ]

COMPUTED_STATE_ATTRIBUTES=[
    'total_loss',
    'loss',
    'acc',
    'best_loss',
    'best_epoch']

STATE_ATTRIBUTES=BASE_STATE_ATTRIBUTES+COMPUTED_STATE_ATTRIBUTES
CB_ERROR='if passing Callbacks instance, default_callbacks should be False'





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
        if criterion and optimizer:
            self.compile(criterion=criterion,optimizer=optimizer)


    def compile(self,criterion,optimizer):
        self.criterion=criterion
        self.optimizer=optimizer


    def fit(self,
            train_loader,
            valid_loader=None,
            nb_epochs=1,
            callbacks=[],
            history_callback=True,           
            noise_reducer=None,
            accuracy_method=None,
            round_prediction=False,
            initial_loss=9e12):
        self._reset_state(nb_epochs=nb_epochs,best_loss=initial_loss)
        self._set_callbacks(
            callbacks,
            history_callback,
            noise_reducer)
        self.accuracy_method=accuracy_method or batch_accuracy(round_prediction)
        self.callbacks.on_train_begin(**self._state())
        for epoch in range(1,nb_epochs+1):
            self._run_epoch(
                epoch=epoch,
                loader=train_loader,
                train_mode=True)
            # validation
            if valid_loader:
                with torch.no_grad():
                    self.callbacks.on_validation_begin(**self._state())
                    self._run_epoch(
                        epoch=epoch,
                        loader=valid_loader,
                        train_mode=False)
                    self.callbacks.on_validation_end(**self._state())
            self.callbacks.on_epoch_end(**self._state())
        self.callbacks.on_train_end(**self._state())


    def _reset_state(self,**kwargs):
        for attr in STATE_ATTRIBUTES:
            setattr(self,attr,kwargs.get(attr,0))


    def _set_callbacks(self,callbacks,history_callback,noise_reducer):
        if isinstance(callbacks,list):
            callbacks=Callbacks(callbacks)
        if history_callback:
            callbacks.append(History(noise_reducer=noise_reducer))
        self.callbacks=callbacks


    def _state(self):
        state={}
        for attr in STATE_ATTRIBUTES:
            state[attr]=getattr(self,attr)       
        return state


    def _update_state(self,**kwargs):
        for attr in BASE_STATE_ATTRIBUTES:
            val = kwargs.get(attr,None)
            if val:
                setattr(self,attr,val)
        if kwargs.get('batch_loss'):
            self.total_loss+=self.batch_loss
            self.loss=self.total_loss/self.batch
        if kwargs.get('batch_acc'):
            self.acc=((self.acc*(self.batch-1))+self.batch_acc)/self.batch


    def _run_epoch(self,
            epoch,
            loader,
            train_mode):
        self._reset_state(epoch=epoch)
        self.callbacks.on_epoch_begin(**self._state())
        for i,batch in enumerate(loader):
            self._run_batch(i+1,batch,train_mode)


    def _run_batch(self,batch_index,batch,train_mode):
        self._update_state(batch=batch_index)
        self.callbacks.on_batch_begin()
        inputs, targets=self._batch_data(batch)
        self.callbacks.on_forward_begin(**self._state())
        outputs=self.model(inputs)
        self.callbacks.on_forward_end(**self._state())
        loss=self.criterion(outputs,targets)
        self._update_state(
            batch_loss=loss.item(),
            batch_acc=self.accuracy_method(outputs,targets))
        self.callbacks.on_loss_computed(**self._state())
        if train_mode:
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


