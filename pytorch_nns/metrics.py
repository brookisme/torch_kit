import torch
import numpy as np
import pytorch_nns.helpers as h
from sklearn.metrics import confusion_matrix


def confusion(pred,targ,categories):
    if not isinstance(categories,int):
        categories=len(categories)
    # TODO: PURE PyTorch Version
    pred=h.to_numpy(pred)
    targ=h.to_numpy(targ)
    cmatrix=confusion_matrix(
        targ.reshape(-1),
        pred.reshape(-1),
        labels=range(categories))
    return cmatrix


def precision(categories,category,cmatrix=None,pred=None,targ=None):
    if cmatrix is None: 
        cmatrix=confusion(pred,targ,categories)
    index=h.get_index(category,categories)
    if (cmatrix[index].sum()==0):
        return 1.0
    else:
        tp_plus_fp=cmatrix[:,index].sum()
    if (tp_plus_fp==0):
        return 0.0 
    else: 
        return cmatrix[index,index]/tp_plus_fp


def recall(categories,category,cmatrix=None,pred=None,targ=None):
    if cmatrix is None: 
        cmatrix=confusion(pred,targ)    
    index=h.get_index(category,categories)
    if (cmatrix[c].sum()==0):
        return 1.0
    else: 
        return cmatrix[c,c]/cmatrix[c].sum()
 

def accuracy(pred,targ):
    return (pred==targ).float().mean()


