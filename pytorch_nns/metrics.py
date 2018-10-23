import torch
import numpy as np
import pytorch_nns.helpers as h
from sklearn.metrics import confusion_matrix

DEFAULT_BETA=2

def target_prediction_argmax(targ,pred,axis=0):
    return h.argmax(pred,axis=axis),h.argmax(targ,axis=axis)


def accuracy(pred,targ,argmax=True,axis=0):
    if argmax:
        targ,pred=target_prediction_argmax(targ,pred,axis=axis)
    return (pred==targ).mean()


def confusion(targ,pred,labels=None,nb_categories=None,argmax=False,axis=0):
    if argmax:
        if not nb_categories:
            nb_categories=targ.shape[axis]
        targ,pred=target_prediction_argmax(targ,pred,axis=axis)
    if not labels:
        labels=range(nb_categories)
    return confusion_matrix(
        targ.reshape(-1),
        pred.reshape(-1),
        labels=labels)


def precision(category_index,cmatrix=None,targ=None,pred=None,argmax=False,axis=0):
    if cmatrix is None:
        cmatrix=confusion(targ,pred,argmax=argmax,axis=axis)
    tp_plus_fp=cmatrix[:,category_index].sum()
    if tp_plus_fp==0:
        return 1.0 
    else: 
        return cmatrix[category_index,category_index]/tp_plus_fp


def recall(category_index,cmatrix=None,targ=None,pred=None,argmax=False,axis=0):
    if not cmatrix:
        cmatrix=confusion(targ,pred,argmax=argmax,axis=axis)
    tp_plus_fn=cmatrix[category_index].sum()
    if tp_plus_fn==0:
        return 1.0
    else:
        return cmatrix[category_index,category_index]/tp_plus_fn


def fbeta(
        precision=None,
        recall=None,
        category_index=None,
        cmatrix=None,
        targ=None,
        pred=None,
        beta=DEFAULT_BETA,
        argmax=False,
        axis=0,
        return_precision_recall=False):
    if not precision:
        cmatrix=confusion(targ,pred,argmax=argmax,axis=axis)
        precision=precision(category_index,cmatrix=cmatrix)
        recall=recall(category_index,cmatrix=cmatrix)
    beta_sq=(beta**2)
    if (precision==0) and (recall==0):
        return 0.0
    else:
        fbeta=(1+beta_sq)*(precision*recall)/(beta_sq*precision + recall)
        if return_precision_recall:
            return fbeta, precision, recall
        else:
            return fbeta



         