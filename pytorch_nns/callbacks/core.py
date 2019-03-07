from .base import Callback

FLOAT_TMPL='{:>.5f}'
ROW_TMPL='{:^10} {:^10} | {:^10} {:^10} | {:^10} {:^10}'


class History(Callback):    
    def __init__(self,noise_reducer=None):
        super(History,self).__init__()
        self.noise_reducer=noise_reducer
        
    
    def on_train_begin(self,**kwargs):
        print('')
        print('-'*75)
        print(ROW_TMPL.format('epoch','batch','batch_loss','loss','batch_acc','acc'))
        print('-'*75)
        self.nb_epochs=kwargs.get('nb_epochs')

        
    def on_batch_end(self,**kwargs):
        self.print_state('on_batch_end',kwargs,flush=False)
        
    
    def on_epoch_end(self,**kwargs):
        self.print_state('on_epoch_end',kwargs,)

        
    def print_state(self,method,kwargs,flush=True):
        if self._print_epoch(kwargs.get('epoch')):
            if flush:
                end=None
            else:
                end='\r'
            print(ROW_TMPL.format(
                kwargs.get('epoch'),
                kwargs.get('batch'),
                self._flt(kwargs.get('batch_loss')),
                self._flt(kwargs.get('loss')),
                self._flt(kwargs.get('batch_acc')),
                self._flt(kwargs.get('acc'))),
            end=end,
            flush=flush)


    def _flt(self,flt):
        return FLOAT_TMPL.format(flt)

    
    def _print_epoch(self,epoch,force=False):
        if force or not self.noise_reducer:
            return True
        else:
            return ((epoch-1)%self.noise_reducer is 0) or epoch==(self.nb_epochs)
