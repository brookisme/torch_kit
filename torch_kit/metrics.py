import torch
import numpy as np
import torch_kit.helpers as h
from sklearn.metrics import confusion_matrix

BETA=2
RETURN_CMATRIX=True
INVALID_ZERO_DIVISON=False
VALID_ZERO_DIVISON=1.0


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
        strict_equality=False,
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
        pred=pred.float()
        targ=targ.float() 
        if strict_equality:
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
    
     
def confusion_matrix(target,prediction,value,ignore_value=None,axis=None):
    true=(target==prediction)
    false=(~true)
    pos=(target==value)
    neg=(~pos)
    keep=(target!=ignore_value)
    tp=(true*pos).sum()
    fp=(false*pos*keep).sum()
    fn=(false*neg*keep).sum()
    tn=(true*neg).sum()
    return H.get_items(tp, fp, fn, tn)


def precision(tp,fp,fn):
    return _precision_recall(tp,fp,fn)


def recall(tp,fn,fp):
    return _precision_recall(tp,fn,fp)


def fbeta(p,r,beta=BETA):
    if p is None: p=precision(tp,fp)
    if r is None: r=recall(tp,fn)
    beta_sq=beta**2
    numerator=(beta_sq*p + r)
    if numerator:
        return (1+beta_sq)*(p*r)/numerator
    else:
        return 0


def stats(
        target,
        prediction,
        value,
        ignore_value=None,
        beta=BETA,
        return_cmatrix=RETURN_CMATRIX):
    tp, fp, fn, tn=confusion_matrix(
        target,
        prediction,
        value,
        ignore_value=ignore_value)
    p=precision(tp,fp,fn)
    r=recall(tp,fn,fp)
    stat_values=[p,r]
    if not _is_false(beta):
        stat_values.append(fbeta(p,r,beta=beta))
    if return_cmatrix:
        stat_values+=[tp, fp, fn, tn]
    return stat_values


#
# INTERNAL
#
def _precision_recall(a,b,c):
    if (a+b):
        return a/(a+b)
    else:
        if c:
            return INVALID_ZERO_DIVISON
        else:
            return VALID_ZERO_DIVISON


def _is_false(value):
    return value in [False,None]






         