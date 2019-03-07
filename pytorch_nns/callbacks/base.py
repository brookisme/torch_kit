import pytorch_nns.helpers as h

""" SKETCH OF TRAINING LOOP
"""
INPT_KEY='input'
TARG_KEY='target'
BASE_STATE_ATTRIBUTES=[
    'epoch',
    'batch',
    'batch_loss',
    'batch_acc' ]

COMPUTED_STATE_ATTRIBUTES=[
    'epoch_total_loss',
    'epoch_loss',
    'epoch_acc' ]

STATE_ATTRIBUTES=BASE_STATE_ATTRIBUTES+COMPUTED_STATE_ATTRIBUTES

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
        self._reset_state()
        if isinstance(callbacks,list):
            callbacks=Callbacks(callbacks)
        self.callbacks=callbacks
        self.callbacks.on_train_begin(**self._state())
        for epoch in range(1,nb_epochs+1):
            self._update_state(epoch=epoch)
            self.callbacks.on_epoch_begin(**self._state())
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


    def _reset_state(self):
        for attr in STATE_ATTRIBUTES:
            setattr(self,attr,0)


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

        self.batch_loss=kwargs.get('batch_loss',0)
        self.epoch_total_loss+=self.batch_loss


    def _run_epoch(self,
            epoch,
            loader,
            train_mode):
        for i,batch in enumerate(loader):
            self._run_batch(i,batch,train_mode)


    def _run_batch(self,batch_index,batch,train_mode):
            self._update_state(batch=batch_index)
            self.callbacks.on_batch_begin()
            inputs, targets=self._batch_data(batch)
            self.callbacks.on_forward_begin(**self._state())
            outputs=self.model(inputs)
            self.callbacks.on_forward_end(**self._state())
            loss=self.criterion(outputs,targets)
            self._update_state(batch_loss=loss.item())
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





""" CALLBACKS
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


        