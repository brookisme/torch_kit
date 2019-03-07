""" SKETCH OF TRAINING LOOP

class Trainer(object):

    '...'


    def fit(self,...):
        callbacks.on_train_begin()
        for epoch in range(1,nb_epochs+1):
            # train
            callbacks.on_epoch_begin()
            self._run_epoch(
                epoch=epoch,
                loader=train_loader,
                train_mode=True)
            # validation
            if valid_loader:
                with torch.no_grad():
                    callbacks.on_validation_begin()
                    self._run_epoch(
                        epoch=epoch,
                        loader=valid_loader,
                        train_mode=False)
                    callbacks.on_validation_end()
            callbacks.on_epoch_end()
        callbacks.on_train_end()



    def _run_epoch(self,
            epoch,
            loader,
            train_mode):
        for batch in loader:
            self._run_batch(batch,train_mode)



    def _run_batch(self,batch,train_mode):
            callbacks.on_batch_begin()
            inputs, targets=self._batch_data(batch)
            callbacks.on_forward_begin()
            outputs=self.model(inputs)
            callbacks.on_forward_end()
            loss=self.criterion(outputs,targets)
            callbacks.on_loss_computed()
            if train_mode:
                self.optimizer.zero_grad()
                callbacks.on_backward_begin()
                loss.backward()
                callbacks.on_backward_end()
                self.optimizer.step()
            callbacks.on_batch_end()
"""



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
            super(Callback,self).__getattr__(attr)

            
            
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
        