import os
import pickle
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import torch_kit.helpers as h
from .base import Callback

TS_FMT="%Y-%m-%dT%H:%M:%S"
FLOAT_TMPL='{:>.5f}'
ROW_TMPL='{:^10} {:^10} | {:^10} {:^10} | {:^10} {:^10}'
DEFAULT_NAME='history'
DEFAULT_DIR='history'
DEFAULT_LOG_DIR='logs'



class History(Callback):

    @staticmethod
    def _init_history():
        return {
            'loss':[],
            'acc':[],
            'batch_loss':[],
            'batch_acc':[]
        }


    def __init__(self,
            save=False,
            name=DEFAULT_NAME,
            history_dir=DEFAULT_DIR,
            noise_reducer=None,
            log=True,
            log_header=True,
            log_dir=DEFAULT_LOG_DIR,
            silent=False):
        super(History,self).__init__()
        self.train_start_timestamp=datetime.now().strftime(TS_FMT)
        self._set_name_path(save,name,history_dir)
        self._set_logger(log,log_header,log_dir)
        self.noise_reducer=noise_reducer
        self.silent=silent
        self.reset_history()


    def on_train_begin(self,**kwargs):
        if not self.silent:
            header=ROW_TMPL.format('epoch','batch','batch_loss','loss','batch_acc','acc')
            self._print('',None,True,True)
            self._print('-'*75,None,True,True)
            self._print(header,None,True,True)
            self._print('-'*75,None,True,True)
            self.nb_epochs=kwargs.get('nb_epochs')


    def on_batch_end(self,**kwargs):
        mode,loss,acc=self._mode_loss_acc(kwargs,is_batch=True)
        self._print_state(mode,kwargs,flush=False)
        self._update_history(
            mode=mode,
            batch_loss=loss,
            batch_acc=acc)
    

    def on_epoch_end(self,**kwargs):
        mode,loss,acc=self._mode_loss_acc(kwargs)        
        self._print_state(mode,kwargs,log=True)
        self._update_history(
            mode=mode,
            loss=loss,
            acc=acc)
        if self.save:
            h.save_pickle(self.history,self.path)


    def on_train_end(self,**kwargs):
        self.train_end_timestamp=datetime.now().strftime(TS_FMT)
        if self.file_handler:
            self.logger.removeHandler(self.file_handler)
            self.logger=None
            self.file_handler=None


    def reset_history(self):
        self.history={
            'train':History._init_history(),
            'valid':History._init_history()
        }    


    def plot(self,batch=False,show=True,figsize=(12,3)):
        plot(self.history,batch=batch,show=show,figsize=figsize)


    #
    #  INTERNEL
    #
    def _set_name_path(self,save,name,history_dir):
        self.save=save
        self.name=name
        if save:
            self.path=f'{name}.{self.train_start_timestamp}.p'
            if history_dir:
                os.makedirs(history_dir,exist_ok=True)
                self.path=f'{history_dir}/{self.path}'


    def _set_logger(self,log,log_header,log_dir):
        if log:
            if isinstance(log,str):
                log_filename=log
            else:
                log_filename=f'{self.name}_{self.train_start_timestamp}.log'
            if log_dir:
                os.makedirs(log_dir,exist_ok=True)
                log_filename=f'{log_dir}/{log_filename}'
            self.file_handler=logging.FileHandler(log_filename)
            self.logger=logging.getLogger(__name__)
            self.logger.addHandler(self.file_handler)
            self.logger.setLevel(logging.DEBUG)
            if log_header:
                if not isinstance(log_header,str):
                    log_header=log_filename
                self.logger.info(log_header)
        else:
            self.logger=False
            self.file_handler=False


    def _print_state(self,mode,kwargs,flush=True,log=False):
        if not self.silent:
            if self._print_epoch(kwargs.get('epoch')):
                msg, end=self._get_log_row(mode,kwargs,flush)
                self._print(msg,end,flush,log)


    def _print(self,msg,end,flush,log):
        if log and self.logger:
            self.logger.info(msg)
        print(
            msg,
            end=end,
            flush=flush)


    def _print_epoch(self,epoch,force=False):
        if force or not self.noise_reducer:
            return True
        else:
            return ((epoch-1)%self.noise_reducer is 0) or epoch==(self.nb_epochs)


    def _mode_loss_acc(self,kwargs,is_batch=False):
        mode=kwargs.get('mode')
        if mode is 'valid':
            prefix='val_'
        else:
            prefix=''
        if is_batch:
            batch_part='batch_'
        else:
            batch_part=''
        loss=kwargs.get(f'{prefix}{batch_part}loss')
        acc=kwargs.get(f'{prefix}{batch_part}acc')
        return mode, loss, acc


    def _get_log_row(self,mode,kwargs,flush):
        if flush:
            end=None
        else:
            end='\r'
        if mode=='train':
            index=kwargs.get('epoch')
            prefix=''
        else:
            index=' - '
            prefix='val_'
        msg=ROW_TMPL.format(
                index,
                kwargs.get('batch'),
                self._flt(kwargs.get(f'{prefix}batch_loss')),
                self._flt(kwargs.get(f'{prefix}loss')),
                self._flt(kwargs.get(f'{prefix}batch_acc')),
                self._flt(kwargs.get(f'{prefix}acc')))
        return msg, end


    def _flt(self,flt):
        return FLOAT_TMPL.format(flt)


    def _update_history(self,
            mode,
            loss=None,
            acc=None,
            batch_loss=None,
            batch_acc=None):
        if loss is not None: self.history[mode]["loss"].append(loss)
        if acc is not None: self.history[mode]["acc"].append(acc)
        if batch_loss is not None: self.history[mode]["batch_loss"].append(batch_loss)
        if batch_acc is not None: self.history[mode]["batch_acc"].append(batch_acc)



#
# HELPER METHODS
#
def _load_pickle(path):
    with open(path,"rb") as f:
        obj=pickle.load(f)
    return obj


def plot(
        hist,
        batch=False,
        show=True,
        figsize=(12,3),
        file_path=None,
        epoch_start=None,
        epoch_end=None,
        **save_kwargs):
    if isinstance(hist,str):
        hist=_load_pickle(hist)
    fig,axs=plt.subplots(1,2,figsize=figsize)
    thist=hist['train']
    vhist=hist['valid']
    if batch:
        loss_key='batch_loss'
        acc_key='batch_acc'
        loss_title='BATCH LOSS'
        acc_title='BATCH ACCURACY'
    else:
        loss_key='loss'
        acc_key='acc' 
        loss_title='LOSS'
        acc_title='ACCURACY'    
    # loss
    plt.legend(loc='best')
    axs[0].set_title(loss_title)
    axs[0].plot(thist.get(loss_key)[epoch_start:epoch_end])
    if vhist:
        axs[0].plot(vhist.get(loss_key)[epoch_start:epoch_end])
    # acc
    axs[1].set_title(acc_title)
    axs[1].plot(thist.get(acc_key)[epoch_start:epoch_end])
    if vhist:
        axs[1].plot(vhist.get(acc_key)[epoch_start:epoch_end])
    if file_path:
        fig.savefig(file_path,**save_kwargs)
    if show:
        plt.show()



