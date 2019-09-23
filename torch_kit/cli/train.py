import os,sys
sys.path.append(os.getcwd())
from importlib import import_module
import torch_kit.train as train
import torch_kit.metrics as metrics

DEFAULT_LRS=[1e-3]
DEFAULT_OPTIMIZER='adam'
CONFIG_ERROR='config should be named (ie { name: config_dict })'

from pprint import pprint

class TrainManager(object):


    def __init__(self,module_name,config):
        self.module=import_module(module_name)
        if len(config)==1:
            self.name=next(iter(config.keys()))
            self.config=next(iter(config.values()))
        else:
            raise ValueError(CONFIG_ERROR)


    def run(self,
            dev=True,
            dry_run=True,
            noise_reducer=None,
            poweroff=False,
            poweroff_wait=30):
        # parse config
        train_loader,valid_loader=self._get('loaders',dev=dev)
        model=self._get('model')
        criterion=self._get('criterion')
        optimizer=self._get('optimizer',DEFAULT_OPTIMIZER)
        lrs=self.config.get('lrs',DEFAULT_LRS)
        # run
        trainer=train.Trainer( model=model, name=self.name )
        trainer.set_callbacks(
            save=True,
            silent=False,
            noise_reducer=noise_reducer,
            name=trainer.name )
        if weights:
            print(f"LOADING WEIGHTS: {weights}")
            trainer.load_weights(weights)
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
                        accuracy_method=metrics.batch_accuracy(pred_argmax=True),
                        nb_epochs=NB_EPOCHS,
                        train_loader=train_loader,
                        valid_loader=valid_loader,
                        early_stopping=True,
                        patience=PATIENCE,
                        patience_start=0)
            self._exit(poweroff,poweroff_wait,dev,dry_run)


    #
    # INTERNAL METHODS
    #
    def _get(self,method_name,config_name=None,**kwargs):
        if not config_name:
            config_name=method_name
        cfig=self.config.get(config_name,{})
        cfig.update(kwargs)
        return getattr(self.module,method_name)(**cfig)


    def _exit(self,poweroff,poweroff_wait,dev,dry_run):
        poweroff=(not dry_run) and (not dev) and poweroff
        print('\n'*10)
        print('*'*100)
        print('\n'*5)
        print('*'*50)
        print('*'*50)
        print('*'*50)
        print('*'*50)
        if poweroff:
            print(f'TURNING OFF COMPUTER IN {poweroff_wait} MINUTES')
            sleep(60*POWEROFF_TIME)
            os.system('sudo poweroff')
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
        if poweroff:
            print('')
            print('...')
            sleep(60*poweroff_wait)
            print('goodbye')
            os.system('sudo poweroff')


