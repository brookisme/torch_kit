import os,sys
sys.path.append(os.getcwd())
from importlib import import_module
from time import sleep
from torchsummary import summary
import torch_kit.train as train
import torch_kit.metrics as metrics
import torch_kit.helpers as h
from . import config as c


DEFAULT_LRS=[1e-3]
DEFAULT_WEIGHTS_DIR='weights'
FORCE_PRED_ARGMAX=999
NB_EPOCHS=50
DEV_NB_EPOCHS=2
PATIENCE=4
CONFIG_ERROR='config should be named (ie { name: config_dict })'
SIZE=256

IS_DEV=c.get('is_dev')
DRY_RUN=c.get('dry_run')
NOISE_REDUCER=c.get('noise_reducer')
POWEROFF=c.get('poweroff')
POWEROFF_WAIT=c.get('poweroff_wait')
PRINT_SUMMARY=c.get('print_summary')


from pprint import pprint


class TrainManager(object):


    def __init__(self,module_name,config,models_config={}):
        self.module=import_module(module_name)
        if len(config)==1:
            self.model_name=next(iter(config.keys()))
            self.config=next(iter(config.values()))
            self.name=f'{module_name}.{self.model_name}'
            self.models_config=models_config
        else:
            raise ValueError(CONFIG_ERROR)


    def run(self,
            dev=IS_DEV,
            dry_run=DRY_RUN,
            noise_reducer=NOISE_REDUCER,
            print_summary=PRINT_SUMMARY):
        if dev: dry_run=False
        # parse config
        train_loader,valid_loader=self._get('loaders',dev=dev)
        model=self._get('model')
        accuracy_activation=self.config.get('criterion',{}).get('accuracy_activation')
        criterion=self._get('criterion')
        optimizer=self._get('optimizer')
        lrs=self.config.get('lrs',DEFAULT_LRS)
        if dev:
            nb_epochs=DEV_NB_EPOCHS
            name=f'DEV---{self.name}'
        else:
            nb_epochs=self.config.get('nb_epochs',NB_EPOCHS)
            name=self.name
        patience=self.config.get('patience',PATIENCE)
        mask_value=self.config.get('mask_value')
        weights=self.config.get('weights')
        pred_argmax=(self._model_config().get('out_ch',FORCE_PRED_ARGMAX)>1)
        # run
        trainer=train.Trainer( model=model, name=name )
        trainer.set_callbacks(
            save=True,
            silent=False,
            noise_reducer=noise_reducer,
            name=trainer.name )
        if weights:
            weights_dir=self.config.get('weights_dir')
            if weights_dir is None:
                weights_dir=DEFAULT_WEIGHTS_DIR
            if weights_dir:
                weights=f'{weights_dir}/{weights}'
            trainer.load_weights(weights)
            if self.config.get('freeze'):
                model=self._get('freeze',model=model)
        if print_summary:
            mcfig=self._model_config()
            size=mcfig.get('size',SIZE)
            in_ch=mcfig['in_ch']
            summary(model.to(h.get_device()),(in_ch,size,size))
        print('\n'*4)
        print('*'*100)
        print('*'*100)
        print('*'*100)
        print()
        print(f"NAME: {self.name}")
        if weights:
            print(f"INIT WEIGHTS: {weights}")
        print(f"LRS: {lrs}")
        if (valid_loader is None) or (valid_loader is False):
            nb_valid=' --- '
        else:
            nb_valid=len(valid_loader)
        print(f"NB_EPOCHS: {nb_epochs}")
        print(f"NB_BATCHES:",len(train_loader),nb_valid)
        print(f"PATIENCE: {patience}")
        print()
        print('='*100)
        print()
        pprint(self.config)
        print()
        print('*'*100)
        print('*'*100)
        print('*'*100)
        print('\n'*4)
        for i,lr in enumerate(lrs):
            print()
            print('-'*100)
            print(f'RUN: {i+1}/{len(lrs)}, LR: {lr}')
            if i: trainer.load_weights()
            if not dry_run:
                trainer.compile(
                    criterion=criterion,
                    optimizer=optimizer(trainer.model.parameters(),lr=lr))
                trainer.fit(
                        accuracy_activation=accuracy_activation,
                        accuracy_method=metrics.batch_accuracy(
                            pred_argmax=pred_argmax,
                            round_prediction=(not pred_argmax),
                            mask_value=mask_value),
                        nb_epochs=nb_epochs,
                        train_loader=train_loader,
                        valid_loader=valid_loader,
                        early_stopping=True,
                        patience=patience,
                        patience_start=0)


    def poweroff(self,poweroff,poweroff_wait,dev,dry_run):
        shutdown=(not dry_run) and (not dev) and poweroff
        print('\n'*10)
        print('*'*100)
        print('\n'*5)
        print('*'*50)
        print('*'*50)
        print('*'*50)
        print('*'*50)
        if shutdown:
            print(f'TURNING OFF COMPUTER IN {poweroff_wait} MINUTES')
        else:
            print(f'DRY_RUN: {dry_run}')
            print(f'DEV: {dev}')
            print(f'POWEROFF: {poweroff}')
            print(f'POWEROFF_WAIT: {poweroff_wait} MINUTES')
        print('*'*50)
        print('*'*50)
        print('*'*50)
        print('*'*50)
        print('\n'*5)
        print('*'*100)
        print('\n'*10)
        if shutdown:
            print('')
            print('...')
            sleep(60*poweroff_wait)
            print('goodbye')
            os.system('sudo poweroff')


    #
    # INTERNAL METHODS
    #
    def _get(self,method_name,config_name=None,**kwargs):
        if not config_name:
            config_name=method_name
        if config_name=='model':
            cfig=self._model_config()
        else:
            cfig=self.config.get(config_name,{})
        if not isinstance(cfig,dict):
            value=cfig
            cfig={}
            cfig[config_name]=value
        cfig.update(kwargs)
        return getattr(self.module,method_name)(**cfig)


    def _model_config(self):
        return self.config.get('model',self.models_config.get(self.model_name,{}))


