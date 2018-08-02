import torch
#
# CONFIG
#
EPS=1e-11


#
# HELPERS
#
def to_tensor(v):
    if isinstance(v,list):
        v=torch.tensor(v)
    elif isinstance(np.ndarray):
        v=torch.from_numpy(v)
    return v


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
    weights=to_tensor(weights).float()
    inpt=inpt/(inpt.sum(1,True)+EPS)
    inpt=torch.clamp(inpt, EPS, 1. - EPS)
    losses=((targ * torch.log(inpt))).float()
    weighted_losses_transpose=weights*losses.transpose(1,-1)
    return -weighted_losses_transpose.mean()



def dice(inpt,targ,weights=None):
    """ (optionally weighted) dice
    
    Args:
        * inpt <tensor>: network prediction 
        * targ <tensor>: network target
        * weights<tensor|nparray|list|None>: optional category weights
    Returns:
        * mean reduction of (weighted) dice
    """    
    if weights:
        weights=to_tensor(weights).float()
    targ=targ.float()
    inpt=(inpt>EPS).float()
    inpt=Variable(inpt, requires_grad=True)
    if weights:
        inpt=(weights*inpt.transpose(1,-1)).transpose(1,-1)
    dice_coefs=2.0*(inpt*targ).sum(1,False)/(inpt+targ).sum(1,False)
    return dice_coefs.mean()
