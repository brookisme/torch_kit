import os,sys
sys.path.append(os.getcwd())
from importlib import import_module


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


    def run(self,dev=True,dry_run=True,noise_reducer=None):
        train_loader,valid_loader=self._get_method('loaders')(
            dev=dev,
            **self.config['loaders'])
        self.train(
                model=self._get_method('model')(**self.config['model']),
                train_loader=train_loader,
                valid_loader=valid_loader,
                criterion=self._get_method('criterion')(**self.config['criterion']),
                optimizer=self._optimizer(**self.config['optimizer']),
                name=self.config.get('name'), 
                lrs=self.config.get('lrs'),
                dry_run=dry_run,
                noise_reducer=noise_reducer)


    def train(self,
            model,
            train_loader,
            valid_loader,
            criterion,
            weights=None,
            name=None,
            optimizer=None,
            lrs=None,
            dry_run=True,
            noise_reducer=None):
        if not name:
            name=DEFAULT_NAME
        if not lrs:
            lrs=DEFAULT_LRS
        if not optimizer:
            optimizer=DEFAULT_OPTIMIZER
        #
        # run
        #
        trainer=train.Trainer( model=model, name=name )
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


    def _get_method(self,method_name):
        print('---',method_name)
        return getattr(self.module,method_name)


    def _optimizer(self,config):
        opt=self._get_method('optimizer')(**config)
        return opt