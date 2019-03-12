from .base import Callback

FLOAT_TMPL='{:>.5f}'
ROW_TMPL='{:^10} {:^10} | {:^10} {:^10} | {:^10} {:^10}'


class History(Callback):

    @staticmethod
    def _init_history():
        return {
            'loss':[],
            'acc':[],
            'batch_loss':[],
            'batch_acc':[]
        }


    def __init__(self,noise_reducer=None,silent=False):
        super(History,self).__init__()
        self.noise_reducer=noise_reducer
        self.silent=silent
        self.reset_history()


    def on_train_begin(self,**kwargs):
        if not self.silent:
            print('')
            print('-'*75)
            print(ROW_TMPL.format('epoch','batch','batch_loss','loss','batch_acc','acc'))
            print('-'*75)
            self.nb_epochs=kwargs.get('nb_epochs')

        
    def on_batch_end(self,**kwargs):
        self.print_state(kwargs.get('mode'),kwargs,flush=False)
        self._update_history(
            mode=kwargs.get('mode'),
            batch_loss=kwargs.get('batch_loss'),
            batch_acc=kwargs.get('batch_acc'))
    

    def on_epoch_end(self,**kwargs):
        self.print_state(kwargs.get('mode'),kwargs)
        self._update_history(
            mode=kwargs.get('mode'),
            loss=kwargs.get('loss'),
            acc=kwargs.get('acc'))

        
    def print_state(self,mode,kwargs,flush=True):
        if not self.silent:
            if self._print_epoch(kwargs.get('epoch')):
                if flush:
                    end=None
                else:
                    end='\r'
                if mode=='train':
                    index=kwargs.get('epoch')
                    prefix=''
                else:
                    index=' - '
                    prefix='val_'
                print(ROW_TMPL.format(
                    index,
                    kwargs.get('batch'),
                    self._flt(kwargs.get(f'{prefix}batch_loss')),
                    self._flt(kwargs.get(f'{prefix}loss')),
                    self._flt(kwargs.get(f'{prefix}batch_acc')),
                    self._flt(kwargs.get(f'{prefix}acc'))),
                end=end,
                flush=flush)


    def reset_history(self):
        self.history={
            'train':History._init_history(),
            'valid':History._init_history()
        }    


    def _flt(self,flt):
        return FLOAT_TMPL.format(flt)

    
    def _update_history(self,
            mode,
            loss=None,
            acc=None,
            batch_loss=None,
            batch_acc=None):
        if loss is not None: self.history[mode]["loss"].append(loss)
        if acc is not None: self.history[mode]["acc"].append(acc)
        if batch_loss is not None: self.history[mode]["batch_loss"].append(batch_loss)
        if batch_acc is not None: self.history[mode]["batch_acc"].append(batch_acc)


    def _print_epoch(self,epoch,force=False):
        if force or not self.noise_reducer:
            return True
        else:
            return ((epoch-1)%self.noise_reducer is 0) or epoch==(self.nb_epochs)
