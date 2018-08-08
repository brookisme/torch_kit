import time
import torch.cuda
import torch.optim as optim
import torch.nn as nn
import pytorch_nns.helpers as h
import pytorch_nns.metrics as metrics
#
# CONFIG
#

INPT_KEY='input'
TARG_KEY='target'
ROW_TMPL='{:^10} {:^10} | {:^10} {:^10} | {:^10} {:^10}'
FLOAT_TMPL='{:>.5f}'

class Trainer(object):


    def __init__(self,model,criterion=None,optimizer=None,force_cpu=False):
        self.device=h.get_device(force_cpu)
        self.model=model.to(self.device)
        if criterion and optimizer:
            self.compile(criterion=criterion,optimizer=optimizer)


    def compile(self,criterion,optimizer):
        self.criterion=criterion
        self.optimizer=optimizer


    def fit(self,train_loader,valid_loader=None,nb_epochs=1,noise_reducer=None):
        # train
        h.print_line("=")
        header=ROW_TMPL.format(
            'Epoch[{}]'.format(nb_epochs),
            'Batch[{}]'.format(len(train_loader)),
            'batch_loss',
            'loss',
            'batch_acc',
            'acc',
            )
        print(header,flush=True)
        for epoch in range(nb_epochs): 
            h.print_line()
            self._run_epoch(
                epoch=epoch,
                loader=train_loader,
                train_mode=True)
            # callback with train end
            # validate
            if valid_loader:
                with torch.no_grad():
                    self._run_epoch(
                        epoch=epoch,
                        loader=valid_loader,
                        train_mode=False)
        h.print_line("=")


    def _run_epoch(self,
            epoch,
            loader,
            train_mode):
        last_index=len(loader)-1
        total_loss=0
        avg_acc=0
        if train_mode:
            self.model.train()
            epoch_index=epoch+1
        else:
            self.model.eval()
            epoch_index="-"
        for i, batch in enumerate(loader):
            log=self._run_batch(batch,train_mode)
            batch_loss=log['loss']
            total_loss+=batch_loss
            avg_loss=total_loss/(i+1)
            batch_acc=log['acc']
            avg_acc=((avg_acc*i)+batch_acc)/(i+1)
            if i==last_index:
                batch_loss='---'
                batch_acc='---'
            else:
                batch_loss=self._flt(batch_loss)
                batch_acc=self._flt(batch_acc)            
            out_row=ROW_TMPL.format(
                epoch_index,
                (i+1),
                batch_loss,
                self._flt(avg_loss),
                batch_acc,
                self._flt(avg_acc))
            print(out_row,end="\r",flush=True)
        print(out_row,flush=True)
        # callback with epoch end


    def _batch_data(self,batch):
        inputs=batch[INPT_KEY].float().to(self.device)
        targets=batch[TARG_KEY].float().to(self.device)
        return inputs, targets


    def _run_batch(self,batch,train_mode):
            inputs, targets=self._batch_data(batch)
            # compute output
            outputs=self.model(inputs)
            loss=self.criterion(outputs,targets)
            log=self._batch_log(loss.item(),outputs,targets)
            # compute gradient and do SGD step
            if train_mode:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return log


    def _batch_log(self,loss,outputs,targets):
        log={}
        log["loss"]=loss
        log["acc"]=metrics.accuracy(outputs,targets)
        return log


    def _flt(self,flt):
        return FLOAT_TMPL.format(flt)

