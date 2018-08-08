import torch
import numpy as np

EPS=1e-8


def _argmax(preds,targs,is_numpy=False):
    if is_numpy:
        preds=np.argmax(preds,axis=1)
        targs=np.argmax(targs,axis=1)
    else:
        preds=torch.argmax(preds,dim=1)
        targs=torch.argmax(targs,dim=1)
    return preds,targs


def _process(preds,targs,value=None,is_numpy=False):
    preds,targs=_argmax(preds,targs,is_numpy=True)
    if value is not None:
        targs=(targs==value)
        preds=(preds==value)
    return preds,targs


def accuracy(preds, targs):
    preds,targs=_argmax(preds,targs)
    return (preds==targs).float().mean()


def accuracy_np(preds, targs):
    preds,targs=_argmax(preds,targs,is_numpy=True)
    return (preds==targs).mean()


def true_positive_count(preds=None,targs=None,value=None,processed=False):
    if not processed: 
        preds,targs=_process(preds,targs,value=value,is_numpy=False)
    return ((preds==targs)*targs).sum()


def true_negatives_count(preds=None,targs=None,value=None,processed=False):
    if not processed: 
        preds,targs=_process(preds,targs,value=value,is_numpy=False)
    return ((preds==targs)*(targs==0)).sum()


def false_positive_count(preds=None,targs=None,value=None,processed=False):
    if not processed: 
        preds,targs=_process(preds,targs,value=value,is_numpy=False)
    return ((preds!=targs)*targs).sum()


def false_negatives_count(preds=None,targs=None,value=None,processed=False):
    if not processed: 
        preds,targs=_process(preds,targs,value=value,is_numpy=False)
    return ((preds!=targs)*(targs==0)).sum()


def precision(preds,targs,value=1):
    preds,targs=_process(preds,targs,value=value,is_numpy=False)
    tp=true_positive_count(preds,targs,processed=True)
    return tp/(preds.sum()+EPS)


def recall(preds,targs,value=1):
    preds,targs=_process(preds,targs,value=value,is_numpy=False)
    tp=true_positive_count(preds,targs,processed=True)
    fn=false_negatives_count(preds,targs,processed=True)
    return tp/(tp+fn+EPS)
