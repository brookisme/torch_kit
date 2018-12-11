import torch
import pandas as pd
import numpy as np
import pytorch_nns.helpers as h
import pytorch_nns.metrics as metrics
import pytorch_nns.utils.multiprocessing as mproc

CATEGORY_NAME_TMPL='c{}'
BAND_NAME_TMPL='b{}'
MAX_PROCESSES=4
PRECISION='precision'
RECALL='recall'
ACCURACY='accuracy'
ACC_IMPORTANCE_TMPL='{}'
PR_IMPORTANCE_TMPL='{}_{}'
EPS=1e-8

class BandImportance(object):
    """
    """
    def __init__(self,
            model,
            dataset,
            measure=PRECISION,
            categories=None,
            bands=None,
            device=None):
        self.model=model.eval()
        self.dataset=dataset
        self.measure=measure
        tmp=dataset[0]
        shape=tmp['input'].shape
        self.band_shape=shape[1:]
        self.nb_pixels=self.band_shape[0]*self.band_shape[1]
        if categories:
            self.nb_categories=len(categories)
            self.categories=categories
        else:
            self.nb_categories=tmp['target'].shape[0]
            self.categories=[
                CATEGORY_NAME_TMPL.format(c) for c in range(self.nb_categories)]
        if bands:
            self.nb_bands=len(bands)
            self.bands=bands
        else:
            self.nb_bands=tmp['input'].shape[0]
            self.bands=[
                CATEGORY_NAME_TMPL.format(c) for c in range(self.nb_bands)]
        self.device=device


    def run(self,max_processes=MAX_PROCESSES,multiprocess=True):
        """
        """
        start,ts_start=h.get_time()
        count=len(self.dataset)
        print("BandImportance.run [{}]:".format(ts_start))
        print("\tinput_count: {}".format(count))
        if multiprocess:
            run_loop=mproc.map_with_threadpool
        else:
            run_loop=mproc.map_sequential
        out=run_loop(
            self._process_prediction,
            range(count),
            max_processes=max_processes)
        end,ts_end=h.get_time()
        print("PredictionMap.run [{}]: complete".format(ts_end))
        print("\tcount: {}".format(len(out)))
        print("\tduration: {}".format(str(end-start)))
        self._process_output(out)


    #
    # INTERNAL METHODS
    #
    def _process_prediction(self,index):
        row=[index]
        data=self.dataset[index]
        inpt=data['input']
        targ=data['target']
        if not torch.is_tensor(inpt):
            inpt=torch.tensor(inpt).unsqueeze(dim=0).float()
            targ=torch.tensor(targ).unsqueeze(dim=0).float()
            if self.device:
                inpt=inpt.to(self.device)
                targ=targ.to(self.device)
        measures=self._measures(inpt,targ)
        row+=measures.tolist()
        for b in range(self.nb_bands):
            rinpt=self._randomize_band(inpt,b)
            rmeasures=self._measures(rinpt,targ)
            b_importances=(measures-rmeasures)/(measures+EPS)
            row+=b_importances.tolist()
        return self._clean_row(row)


    def _process_output(self,out):
        # rows=list(zip(*out))
        rows=list(out)
        columns=['index']+self._importance_columns()
        self.r=rows
        self.c=columns
        self.importance=pd.DataFrame(rows,columns=columns)


    def _randomize_band(self,inpt,band_index):
        idx=torch.randperm(self.nb_pixels)
        band=inpt[:,band_index]
        band=band.view(-1)[idx].view(self.band_shape)
        inpt[:,band_index]=band
        return inpt


    def _measures(self,inpt,targ):
        pred=self.model(inpt)
        pred=h.argmax(pred[0])                
        targ=h.argmax(targ[0])   
        if self.measure==ACCURACY:
            return self._accuracy(pred,targ)
        elif self.measure==RECALL:
            return self._recalls(pred,targ)
        else:
            return self._precisions(pred,targ)
        

    def _accuracy(self,pred,targ):
        return np.array([ metrics.accuracy(pred,targ) ])


    def _recalls(self,pred,targ):
        cm=metrics.confusion(pred,targ,self.nb_categories)
        return np.array(
            [ metrics.recall(self.categories,c,cm) for c in self.categories ] )


    def _precisions(self,pred,targ):
        cm=metrics.confusion(pred,targ,self.nb_categories)
        self.cm=cm
        return np.array(
            [ metrics.precision(self.categories,c,cm) for c in self.categories ] )


    def _importance_columns(self):
        if self.measure==ACCURACY:
            return ['accuracy']+[ ACC_IMPORTANCE_TMPL.format(b) for b in self.bands ]
        else:
            cols=[ f'{c}_{self.measure}' for c in self.categories ]
            for b in self.bands:
                for c in self.categories:
                    cols.append( PR_IMPORTANCE_TMPL.format(b,c) )
            return cols


    def _clean_row(self,row):
        return np.where(
                np.isnan(row),
                np.nan,
                np.where(np.array(row)<0,0,row))

