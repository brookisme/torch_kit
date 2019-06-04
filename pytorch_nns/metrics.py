import torch
import numpy as np
import pytorch_nns.helpers as h
from sklearn.metrics import confusion_matrix

DEFAULT_BETA=2
VALIDITY_BOUND=1e-3
INVALID_VALUE=False
ZERO_DENOMINATOR_VALUE=1.0

import torch
def u(a):
    try:
        return np.unique(a)
    except:
        return torch.unique(a)

def accuracy(
        pred,
        targ,
        argmax=False,
        pred_argmax=False,
        round_prediction=False,
        true_threshold=0.5,
        axis=0,
        mask_value=None):
    if argmax:
        targ=h.argmax(targ,axis=axis)        
        pred=h.argmax(pred,axis=axis)
    elif pred_argmax:
        pred=h.argmax(pred,axis=axis)
    elif round_prediction:
        shift=true_threshold-0.5
        pred=pred+shift
        pred=torch.round(pred)
    if argmax or pred_argmax or round_prediction:
        pred=pred.long()
        targ=targ.long()
        test=(pred==targ)
    else:
        test=1-torch.pow((pred-targ),2)
    if mask_value is not None:
        msk=(targ!=mask_value)
        test=test[msk]
    if torch.is_tensor(test):
        test=test.float()
    return test.mean()


def batch_accuracy(
        pred_argmax=False,
        round_prediction=False,
        mask_value=None,
        argmax=None):
    if argmax is None:
        argmax=((not round_prediction) and (not pred_argmax))
    def _calc(outputs,targets):
        return accuracy(
            outputs,
            targets,
            pred_argmax=pred_argmax,
            argmax=argmax,
            round_prediction=round_prediction,
            mask_value=mask_value,
            axis=1)
    return _calc
    

def confusion(targ,pred,labels=None,nb_categories=None,argmax=False,axis=0):
    if argmax:
        if not nb_categories:
            nb_categories=targ.shape[axis]
        targ,pred=target_prediction_argmax(targ,pred,axis=axis)
    if labels is None:
        labels=range(nb_categories)
    return confusion_matrix(
        targ.reshape(-1),
        pred.reshape(-1),
        labels=labels)


def precision(
        category_index,
        cmatrix=None,
        targ=None,
        pred=None,
        argmax=False,
        axis=0,
        total=None,
        validity_bound=VALIDITY_BOUND):
    if cmatrix is None:
        cmatrix=confusion(targ,pred,argmax=argmax,axis=axis)
    return _valid_divider(
        validity_bound,
        cmatrix[category_index,category_index],
        cmatrix[:,category_index].sum(),
        total or cmatrix.sum())


def recall(
        category_index,
        cmatrix=None,
        targ=None,
        pred=None,
        argmax=False,
        axis=0,
        total=None,
        validity_bound=None):
    if cmatrix is None:
        cmatrix=confusion(targ,pred,argmax=argmax,axis=axis)
    return _valid_divider(
        validity_bound,
        cmatrix[category_index,category_index],
        cmatrix[category_index].sum(),
        total or cmatrix.sum())


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
        total=None,
        validity_bound=None,
        return_precision_recall=False):
    if precision is None:
        cmatrix=confusion(targ,pred,argmax=argmax,axis=axis)
        precision=precision(
            category_index=category_index,
            cmatrix=cmatrix,
            zero_bound=zero_bound,
            zero_bound_value=zero_bound_value)
        recall=recall(
            category_index=category_index,
            cmatrix=cmatrix,
            zero_bound=zero_bound,
            zero_bound_value=zero_bound_value)
    if _is_false(precision) or _is_false(recall):
        return INVALID_VALUE
    else:
        beta_sq=(beta**2)
        if (precision==0) and (recall==0):
            return 0.0
        else:
            fbeta=(1+beta_sq)*(precision*recall)/(beta_sq*precision + recall)
            if return_precision_recall:
                return fbeta, precision, recall
            else:
                return fbeta


def _valid_divider(validity_bound,numerator,denominator,total):
        if denominator==0:
            return ZERO_DENOMINATOR_VALUE
        elif numerator:
            return numerator/denominator
        elif validity_bound and _is_not_valid(
                validity_bound,
                denominator,
                total):
            return INVALID_VALUE
        else:
            return 0


def _is_not_valid(validity_bound,value,total):
    return ((value/total)<validity_bound)


def _is_false(value):
    return value in [False,None]






         