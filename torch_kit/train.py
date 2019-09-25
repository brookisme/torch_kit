import os
from datetime import datetime
import torch
import torch.nn as nn
import torch_kit.helpers as h
import torch_kit.metrics as metrics
from torch_kit.callbacks import Callbacks, History



#
# CONSTANTS
#
INPT_KEY='input'
TARG_KEY='target'
DEFAULT_NAME='train'
WEIGHTS_DIR='weights'
TS_FMT="%Y-%m-%dT%H:%M:%S"
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
DOUBLE='DoubleTensor'
SKIPPED='skipped'
WRITE_REPORT=True
REPORTS_DIR='reports'

#
# TRAINER
#
class Trainer(object):
    """ pytorch training loop

    Usage:
        * create instance: trn=Trainer(...) 
        * set_callbacks: trn.set_callbacks(...)
        * train: trn.fit(...)
        * load best weights to model: trn.load_weights()
        * plot history: trn.callbacks.get('history').plot()

    Args:
        model<nn.Module>: pytorch model
        criterion<criterion|None>: 
            - pytorch criterion
            - can be set later with .compile(...)
            - if criterion and optimizer are provide .compile() is automatically called
        optimizer<optimizer|None>: 
            - pytorch optimizer
            - can be set later with .compile(...)
            - if criterion and optimizer are provide .compile() is automatically called
        name<str>: weights are saved to files containing this name and a timestamp
        weights_dir<str>: 
            - directory for weights
            - defaults to `weights/` in current directory
        force_cpu<bool>: if true run on cpu even if gpu exists
    """ 
    def __init__(self,
            model,
            criterion=None,
            optimizer=None,
            name=DEFAULT_NAME,
            weights_dir=WEIGHTS_DIR,
            reports_dir=REPORTS_DIR,
            write_report=WRITE_REPORT,
            force_cpu=False):
        self.device=h.get_device(force_cpu)
        self.model=model.to(self.device)
        self.name=name
        self.weights_dir=weights_dir
        self.reports_dir=reports_dir
        self.write_report=write_report
        self.callbacks=False
        self.best_weights_path=None
        if criterion and optimizer:
            self.compile(criterion=criterion,optimizer=optimizer)


    def compile(self,criterion=None,optimizer=None):
        if criterion: self.criterion=criterion
        if optimizer: self.optimizer=optimizer


    def set_callbacks(self,
            callbacks=[],
            force=False,
            history_callback=True,
            **history_kwargs):
        if force or (self.callbacks is False):
            if isinstance(callbacks,list):
                callbacks=Callbacks(callbacks)
            if history_callback:
                callbacks.add(History(**history_kwargs))
            self.callbacks=callbacks
        else:
            raise ValueError(CALLBACK_ERROR)


    def report(self,write=False):
        if write:
            os.makedirs(self.reports_dir,exist_ok=True)
            name=f"{self.reports_dir}/report.{self.name}.{self.timestamp}.txt"
            file=open(name,"w")
        else:
            file=False
        if self.best_weights_path:
            self._print(f"Trainer.{self.name}.{self.timestamp}:",file)
            self._print(f"\t best_epoch: {self.best_epoch}",file)
            self._print(f"\t best_loss: {self.best_loss}",file)
            self._print(f"\t best_acc: {self.best_acc}",file)
            if self.save_best:
                self._print(f"\t {self.best_weights_path}",file)
            if self.save_all:
                self._print(f"\t {self.weights_path}",file)
        if file: file.close() 
        else:
            print("Trainer: No training to report")


    def save_weights(self,
            name=None,
            path=None,
            timestamp=True,
            tag=None,
            noisy=True):
        path=self._build_path(self.weights_dir,name,path,tag,timestamp)
        if noisy: print(f"Trainer.save_weights: {path}")
        torch.save(self.model.state_dict(),path)
        return path


    def load_weights(self,path=None,noisy=True):
        if not path:
            path=self.best_weights_path
        if path:
            if noisy:
                print(f"Trainer.loading_weights: {path}")
            h.load_weights(
                self.model,
                path,
                device=self.device)
        else:
            print(f"[WARN] Trainer.loading_weights: no weights to be loaded. skipping")


    def fit(self,
            train_loader,
            valid_loader=None,
            valid_reducer=None,
            nb_epochs=1,
            accuracy_method=None,
            accuracy_activation=None,
            round_prediction=False,
            early_stopping=False,
            patience=0,
            patience_start=0,
            save_best=True,
            save_all=False,
            initial_loss=9e12):
        self.timestamp=self._get_timestamp()
        self._reset_state(nb_epochs=nb_epochs)
        self.best_loss=initial_loss
        self.valid_reducer=valid_reducer
        self.best_acc=-1
        self.best_epoch=0
        self.save_best=save_best
        self.save_all=save_all
        self.early_stopping=early_stopping
        self.patience=patience
        self.patience_start=patience_start
        self.patience_count=0
        self.accuracy_method=accuracy_method or metrics.batch_accuracy(round_prediction)
        self.accuracy_activation=self._get_accuracy_activation(accuracy_activation)
        self.callbacks.on_train_begin(**self._state())
        for epoch in range(1,nb_epochs+1):
            self._reset_state(epoch=epoch)
            self._run_epoch(
                epoch=epoch,
                loader=train_loader,
                mode='train')
            # validation
            if valid_loader:
                if (not self.valid_reducer) or ((epoch-1)%self.valid_reducer is 0):
                    with torch.no_grad():
                        self.callbacks.on_validation_begin(**self._state())
                        self._run_epoch(
                            epoch=epoch,
                            loader=valid_loader,
                            mode='valid')
                        self.callbacks.on_validation_end(**self._state())
                    is_best=self._check_for_best(
                        epoch=epoch,
                        loss=self.val_loss,
                        acc=self.val_acc)
                else:
                    is_best=SKIPPED
            else:
                is_best=self._check_for_best(epoch=epoch,loss=self.loss,acc=self.acc)
            if is_best:
                if self.save_best:
                    if is_best is True:
                        self.best_weights_path=self.save_weights(tag='best',noisy=False)
                    else:
                        self.save_weights(tag='skip',noisy=False)
            elif self.early_stopping:
                if not self._check_patience(epoch):
                    break
            if self.save_all:
                self.weights_path=self.save_weights(noisy=False)
            self.callbacks.on_epochs_complete(**self._state())
        self.callbacks.on_train_end(**self._state())
        self.report(self.write_report)


    #
    #  INTERNAL
    #
    def _reset_state(self,**kwargs):
        for attr in STATE_ATTRIBUTES:
            setattr(self,attr,kwargs.get(attr,0))


    def _get_accuracy_activation(self,act):
        if isinstance(act,str):
            act=act.lower()
            if act=='sigmoid':
                act=nn.Sigmoid()
            elif act=='softmax':
                act=nn.Softmax()
        elif act is True:
            act=nn.Sigmoid()
        return act


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
        if mode=='train':
            self.model=self.model.train()
        else:
            self.model=self.model.eval()
        self._update_state(mode=mode)
        self.callbacks.on_epoch_begin(**self._state())
        for i,batch in enumerate(loader):
            self._run_batch(i+1,batch,mode)
        self.callbacks.on_epoch_end(**self._state())


    def _run_batch(self,batch_index,batch,mode):
        self._update_state(batch=batch_index)
        self.callbacks.on_batch_begin(**self._state())
        inputs, targets=self._batch_data(batch)
        self.callbacks.on_forward_begin(**self._state())
        outputs=self.model(inputs)
        self.callbacks.on_forward_end(**self._state())
        loss=self.criterion(outputs,targets)
        if self.accuracy_activation:
            outputs=self.accuracy_activation(outputs)
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
        inputs=batch[INPT_KEY].to(self.device)
        targets=batch[TARG_KEY].to(self.device)
        if DOUBLE in inputs.type():
            inputs=inputs.float()
        if DOUBLE in targets.type():
            targets=targets.float()
        return inputs, targets


    def _check_for_best(self,epoch,loss,acc):
        if self.best_loss>loss:
            self.best_epoch=epoch
            self.best_loss=loss
            self.best_acc=acc
            self.patience_count=0
            return True
        else:
            return False


    def _check_patience(self,epoch):
        if epoch>=self.patience_start:
            self.patience_count+=1
        return self.patience>=self.patience_count


    def _build_path(self,directory,name,path,tag,timestamp):
        if not path:
            parts=[]
            if directory:
                parts.append(directory)
                os.makedirs(directory,exist_ok=True)
            parts.append(name or self.name)
            path=os.path.join(*parts)
            if tag:
                path=f'{path}.{tag}'
            if timestamp:
                path=f'{path}.{self.timestamp}'
            path=f'{path}.p'
        return path


    def _print(self,msg,file=False):
        print(msg)
        if file:
            file.write(f"{msg}\n")


    def _get_timestamp(self,none=False):
        return datetime.now().strftime(TS_FMT)


