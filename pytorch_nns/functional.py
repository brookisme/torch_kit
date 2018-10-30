import math
import torch
import pytorch_nns.helpers as h
import numpy as np
#
# CONFIG
#
EPS=1e-9



#
# HELPERS
#
def category_weights(
        count_dict,
        total=None,
        use_log=True,
        multiplier=0.15,
        max_weight=5.0):
    """ category_weights

    Args:
        * count_dict <dict>: dictionary of category counts 
        * total <int|None>: total count (if None compute)
        * use_log <bool [True]>: take log of distribution weight
        * multiplier <float>: multiplier for log argument
        * min_weight <float [1.0]>: min weight value
    Returns:
        * mean reduction of weighted categorical crossentropy
    """
    weights={}
    if not total:
        total=sum(list(count_dict.values()))
    for key in count_dict.keys():
        v=count_dict[key]
        if not v: v=EPS
        weight=multiplier*total/float(v)
        if use_log:
            weight=math.log(weight)
        weights[key]=weight
    minv=min(weights.values())
    weights={ k: min(max_weight,v/minv) for k,v in weights.items() }
    return weights




#
# LOSSES
#
def weighted_categorical_crossentropy(inpt,targ,weights):
    """ weighted_categorical_crossentropy

    Args:
        * inpt <tensor>: network prediction 
        * targ <tensor>: network target
        * weights<tensor|nparray|list>: category weights
    Returns:
        * mean reduction of weighted categorical crossentropy
    """
    weights=h.to_tensor(weights).float()
    inpt=inpt/(inpt.sum(1,True)+EPS)
    inpt=torch.clamp(inpt, EPS, 1. - EPS)
    losses=((targ * torch.log(inpt))).float()
    weighted_losses_transpose=weights*losses.transpose(1,-1)
    return -weighted_losses_transpose.mean()*targ.size(1)


def dice(inpt,targ,weights=None):
    """ (optionally weighted) dice
    
    Args:
        * inpt <tensor>: network prediction 
        * targ <tensor>: network target
        * weights<tensor|nparray|list|None>: optional category weights
    Returns:
        * mean reduction of (weighted) dice
    """    
    if weights is not None:
        weights=h.to_tensor(weights).float()
    targ=targ.float()
    inpt=(inpt>EPS).float()
    inpt.requires_grad=True
    if weights is not None:
        inpt=(weights*inpt.transpose(1,-1)).transpose(1,-1)
    dice_coefs=2.0*(inpt*targ).sum(1,False)/(inpt+targ+EPS).sum(1,False)
    return -dice_coefs.mean()


"""
def soft_dice_loss(y_pred, y_true, epsilon=1e-6): 
    ''' TODO: COMPARE MY DICE TO THIS

    https://www.jeremyjordan.me/semantic-segmentation/#loss
    
    ...

    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
  
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    
    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''
    
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
    return 1 - np.mean(numerator / (denominator + epsilon)) # average over classes and batch
"""
