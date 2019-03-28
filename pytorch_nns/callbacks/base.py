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
        'on_epochs_complete',
        'on_train_end',
    ]
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
        self.callbacks=self._to_dict(callbacks)
        for m in Callback.METHODS:
            self._set_method(m)


    def add(self,callback,name=None):
        if not name:
            try:
                name=callback.name
            except:
                name=len(self.callbacks)
        self.callbacks[name]=callback


    def get(self,key,default=None):
        return self.callbacks.get(key,default)


    def list(self):
        return list(self.callbacks.keys())


    def __getitem__(self, index):
        return self.callbacks[index]


    def _set_method(self,m):
        def cb_method(**kwargs):
            for _,cb in self.callbacks.items():
                getattr(cb,m)(**kwargs)
        setattr(self,m,cb_method)


    def _to_dict(self,callbacks):
        cbks={}
        for i,cb in enumerate(callbacks):
            if isinstance(cb,dict):
                k=next(iter(cb.keys()))
                v=next(iter(cb.values()))
            else:
                try:
                    k,v=cb.name,cb
                except:
                    k,v=i,cb
            cbks[k]=v
        return cbks


        