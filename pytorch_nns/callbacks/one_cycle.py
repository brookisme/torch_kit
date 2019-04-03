import os
import logging
from datetime import datetime
import pytorch_nns.helpers as h
from .base import Callback

TS_FMT="%Y-%m-%dT%H:%M:%S"
FLOAT_TMPL='{:>.5f}'
ROW_TMPL='{:^10} {:^10} | {:^10} {:^10} | {:^10} {:^10}'
DEFAULT_NAME='1cycle'
DEFAULT_LOG_DIR='logs'

class OneCycle(Callback):


    def __init__(
            self,
            optimizer,
            nb_batches,
            max_lr=1e-2,
            out_pct=5,
            init_reduction=10,
            min_reduction=100,
            cycle_momentum=True,
            max_momentum=0.95,
            min_momentum=0.85,
            momentum_key='betas',
            momentum_index=0,
            name=DEFAULT_NAME):
        super(OneCycle,self).__init__()
        self.optimizer=optimizer
        self.halfcycle_len=int(nb_batches*(1-out_pct/100)/2)
        self.out_len=nb_batches-2*self.halfcycle_len
        self.max_lr=max_lr
        self.init_lr=max_lr/init_reduction
        self.min_lr=max_lr/min_reduction
        self.cycle_momentum=cycle_momentum
        self.max_momentum=max_momentum
        self.min_momentum=min_momentum
        self.momentum_key=momentum_key
        self.momentum_index=momentum_index
        self.name=name


    def on_batch_begin(self,**kwargs):
        lr=self._get_lr(kwargs['batch'])
        self._update_lr(lr)
        if self.cycle_momentum:
            mom=self._get_mom(kwargs['batch'])
            self._update_mom(mom)


    def _get_lr(self,batch):
        if batch<self.halfcycle_len:
            steps=self.halfcycle_len
            y_in=self.init_lr
            y_out=self.max_lr
            step=batch
        elif batch<2*self.halfcycle_len:
            steps=self.halfcycle_len
            y_in=self.max_lr
            y_out=self.init_lr
            step=batch-self.halfcycle_len
        else:
            steps=self.out_len-1
            y_in=self.init_lr
            y_out=self.min_lr
            step=batch-2*self.halfcycle_len
        return self._line(y_in,y_out,steps,step)


    def _get_mom(self,batch):
        if batch<self.halfcycle_len:
            y_in=self.max_momentum
            y_out=self.min_momentum
            step=batch
            return self._line(y_in,y_out,self.halfcycle_len,step)
        elif batch<2*self.halfcycle_len:
            y_in=self.min_momentum
            y_out=self.max_momentum
            step=batch-self.halfcycle_len
            return self._line(y_in,y_out,self.halfcycle_len,step)
        else:
            return self.max_momentum


    def _line(self,y_in,y_out,steps,step):
        return y_in+(step-1)*(y_out-y_in)/steps


    def _update_lr(self,lr):
        for grp in self.optimizer.param_groups:
            grp['lr']=lr


    def _update_mom(self,mom):
        for grp in self.optimizer.param_groups:
            if isinstance(self.momentum_index,int):
                grp[self.momentum_key][self.momentum_index]=mom
            else:
                grp[self.momentum_key]=mom

