import os.path
from datetime import datetime
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
HISTORY_DIR='history'
WEIGHTS_DIR='weights'
HISTORY_NAME='history'
WEIGHTS_NAME='weights'
HISTORY='HISTORY'
WEIGHTS='WEIGHTS'
HISTORY_DICT={
    'loss':[],
    'acc':[],
    'batch_loss':[],
    'batch_acc':[]
}



class Trainer(object):


    def __init__(self,
            model,
            criterion=None,
            optimizer=None,
            force_cpu=False,
            name=None,
            history_dir=HISTORY_DIR,
            weights_dir=WEIGHTS_DIR,
            noise_reducer=None):
        self.device=h.get_device(force_cpu)
        self.model=model.to(self.device)
        self.name=name
        self.history_dir=history_dir
        self.weights_dir=weights_dir
        self.noise_reducer=noise_reducer
        if criterion and optimizer:
            self.compile(criterion=criterion,optimizer=optimizer)
        self.reset_history()


    def compile(self,criterion,optimizer):
        self.criterion=criterion
        self.optimizer=optimizer


    def reset_history(self):
        self.history={
            'train':dict(HISTORY_DICT),
            'valid':dict(HISTORY_DICT)
        }


    def save_history(self,
            name=None,
            path=None,
            timestamp=True,
            absolute_path=False,
            ext='p'):
        return self._save_obj(
                self.history,
                HISTORY,
                name=name,
                path=path,
                timestamp=timestamp,
                absolute_path=absolute_path,
                ext='p')



    def save_weights(self,
            name=None,
            path=None,
            timestamp=True,
            absolute_path=False,
            ext='p'):
        return self._save_obj(
                self.model.state_dict(),
                WEIGHTS,
                name=name,
                path=path,
                timestamp=timestamp,
                absolute_path=absolute_path,
                ext='p')


    def fit(self,train_loader,valid_loader=None,nb_epochs=1):
        # train
        h.print_line("=")
        if valid_loader:
            batch_head='B[{},{}]'.format(len(train_loader),len(valid_loader))
        else:
            batch_head='B[{}]'.format(len(train_loader))
        header=ROW_TMPL.format(
            'E[{}]'.format(nb_epochs),
            batch_head,
            'batch_loss',
            'loss',
            'batch_acc',
            'acc',
            )
        print(header,flush=True)
        for epoch in range(nb_epochs):
            print_epoch=self._print_epoch(epoch,nb_epochs)
            if print_epoch: h.print_line()
            self._run_epoch(
                epoch=epoch,
                loader=train_loader,
                train_mode=True,
                print_epoch=print_epoch)
            # callback with train end
            # validate
            if valid_loader:
                with torch.no_grad():
                    self._run_epoch(
                        epoch=epoch,
                        loader=valid_loader,
                        train_mode=False,
                        print_epoch=print_epoch)
        h.print_line("=")


    def _run_epoch(self,
            epoch,
            loader,
            train_mode,
            print_epoch):
        last_index=len(loader)-1
        total_loss=0
        avg_acc=0
        if train_mode:
            self.model.train()
            epoch_index=epoch+1
        else:
            self.model.eval()
            if print_epoch:
                epoch_index="-"
            else:
                epoch_index="t{}".format(epoch+1)
        for i, batch in enumerate(loader):
            log=self._run_batch(batch,train_mode)
            batch_loss=log['loss']
            total_loss+=batch_loss
            avg_loss=total_loss/(i+1)
            batch_acc=log['acc']
            avg_acc=((avg_acc*i)+batch_acc)/(i+1)
            self._update_history(
                train_mode=train_mode,
                batch_loss=batch_loss,
                batch_acc=batch_acc)
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
        self._update_history(
            train_mode=train_mode,
            loss=avg_loss,
            acc=avg_acc)
        if print_epoch: print(out_row,flush=True)


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


    def _update_history(self,
            train_mode,
            loss=None,
            acc=None,
            batch_loss=None,
            batch_acc=None):
        if train_mode:
            mode_key='train'
        else:
            mode_key='valid'
        if loss is not None: self.history[mode_key]["loss"].append(loss)
        if acc is not None: self.history[mode_key]["acc"].append(acc)
        if batch_loss is not None: self.history[mode_key]["batch_loss"].append(batch_loss)
        if batch_acc is not None: self.history[mode_key]["batch_acc"].append(batch_acc)


    def _build_path(self,path_type,name,path,absolute_path,timestamp,ext):
        if not absolute_path:
            if path_type==HISTORY:
                root_dir=self.history_dir
                default_name=HISTORY_NAME
            else:
                root_dir=self.weights_dir
                default_name=WEIGHTS_NAME
            name=name or self.name or default_name
            parts=[p for p in [root_dir,path,name] if p is not None]
            path=os.path.join(*parts)
            if timestamp:
                path="{}.{}".format(
                    path,
                    datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
            if ext:
                path="{}.{}".format(path,ext)
        return path


    def _print_epoch(self,epoch,nb_epochs):
        if self.noise_reducer is None:
            return True
        else:
            return (epoch%self.noise_reducer is 0) or epoch==(nb_epochs-1)


    def _save_obj(self,
            obj,
            obj_name,
            name=None,
            path=None,
            timestamp=True,
            absolute_path=False,
            ext='p'):
        path=self._build_path(obj_name,name,path,absolute_path,timestamp,ext)
        print("Trainer.save_{}:".format(obj_name.lower()),path)
        obj_dir=os.path.dirname(path)
        if obj_dir:
            os.makedirs(obj_dir,exist_ok=True)
        torch.save(obj,path)
        return path






