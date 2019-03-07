""" SKETCH OF TRAINING LOOP

INPT_KEY='input'
TARG_KEY='target'

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
            callbacks=[],
            nb_epochs=1):
        if isinstance(callbacks,list):
            callbacks=cb.Callbacks(callbacks)
        self.callbacks=callbacks
        self.callbacks.on_train_begin()
        for epoch in range(1,nb_epochs+1):
            # train
            self.callbacks.on_epoch_begin()
            self._run_epoch(
                epoch=epoch,
                loader=train_loader,
                train_mode=True)
            # validation
            if valid_loader:
                with torch.no_grad():
                    self.callbacks.on_validation_begin()
                    self._run_epoch(
                        epoch=epoch,
                        loader=valid_loader,
                        train_mode=False)
                    self.callbacks.on_validation_end()
            self.callbacks.on_epoch_end()
        self.callbacks.on_train_end()



    def _run_epoch(self,
            epoch,
            loader,
            train_mode):
        for batch in loader:
            self._run_batch(batch,train_mode)


    def _run_batch(self,batch,train_mode):
            self.callbacks.on_batch_begin()
            inputs, targets=self._batch_data(batch)
            self.callbacks.on_forward_begin()
            outputs=self.model(inputs)
            self.callbacks.on_forward_end()
            loss=self.criterion(outputs,targets)
            self.callbacks.on_loss_computed(loss=loss)
            if train_mode:
                self.optimizer.zero_grad()
                self.callbacks.on_backward_begin()
                loss.backward()
                self.callbacks.on_backward_end()
                self.optimizer.step()
            self.callbacks.on_batch_end(loss=loss)

            
    def _batch_data(self,batch):
        print(type(batch))
        inputs=batch[INPT_KEY].float().to(self.device)
        targets=batch[TARG_KEY].float().to(self.device)
        return inputs, targets
        
"""


ATTRIBUTE_ERROR="'{}' object has no attribute '{}'"
class Callback(object):
    

    METHODS=[
        'on_train_begin',
        'on_epoch_begin',
        'on_batch_begin',
        'on_forward_begin',
        'on_forward_end',
        'on_loss_computed',
        'on_backward_begin',
        'on_backward_end',
        'on_batch_end',
        'on_validation_begin',
        'on_validation_end',
        'on_epoch_end',
        'on_train_end',
    ]
    

    def __init__(self,**kwargs):
        self._set_properites(kwargs)
        
        
    def _set_properites(self,kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)

            
    def __getattr__(self,attr):
        if attr in Callback.METHODS:
            return (lambda **kw: None)
        else:
            raise AttributeError(
                    ATTRIBUTE_ERROR.format(
                        self.__class__.__name__,
                        attr ))



            
            
class Callbacks(object):
    
    def __init__(self,callbacks):
        self.callbacks=callbacks
        for m in Callback.METHODS:
            self._set_method(m)


    def _set_method(self,m):
        def cb_method(**kwargs):
            for cb in self.callbacks:
                getattr(cb,m)(**kwargs)
        setattr(self,m,cb_method)


        