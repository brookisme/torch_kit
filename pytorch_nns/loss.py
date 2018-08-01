import torch.nn as nn
from torch.autograd import Variable, Function
#
# CONFIG
#
EPS=1e-11


#
# HELPERS
#
def _to_tensor(v):
    if isinstance(v,list):
        v=torch.tensor(v)
    elif isinstance(np.ndarray):
        v=torch.from_numpy(v)
    return v




#**********************************************************************
#
# WEIGHTED CATEGORICAL CROSS ENTROPY
#
#**********************************************************************
def weighted_categorical_crossentropy(weights):
    """ weighted_categorical_crossentropy

        Args:
            * weights<tensor|np.array|list>: crossentropy weights
        Returns:
            * weighted categorical crossentropy function
    """
    weights=_to_tensor(weights)
    def loss(inpt,targ):
        targ=torch.tensor(targ)
        inpt=torch.tensor(inpt)
        inpt/=(inpt.sum(1,True)+EPS)
        inpt=torch.clamp(inpt, EPS, 1. - EPS)
        losses=((targ * torch.log(inpt)))
        weighted_losses_transpose=(weights.float()*losses.transpose(1,-1).float())
        return -weighted_losses_transpose.mean()
    return loss


class WeightedCategoricalCrossentropy(nn.Module):
    """ weighted_categorical_crossentropy

        Args:
            * weights<ktensor|nparray|list>: crossentropy weights
        Returns:
            * weighted categorical crossentropy function
    """
    def __init__(self, weights):
        super(WeightedCategoricalCrossentropy, self).__init__()
        self.loss=weighted_categorical_crossentropy(weights)

        
    def forward(self, inpt, targ):
        return self.loss(inpt,targ)




#**********************************************************************
#
# WEIGHTED DICE
#
#**********************************************************************
def dice(weights=None):
    if weights:
        weights=_to_tensor(weights).float()
    def loss(inpt, targ):
        targ=targ.float()
        inpt=(inpt>EPS).float()
        inpt=Variable(inpt, requires_grad=True)
        if weights:
            inpt=(weights*inpt.transpose(1,-1)).transpose(1,-1)
        dice_coefs=2.0*(inpt*targ).sum(1,False)/(inpt+targ).sum(1,False)
        return dice_coefs.mean()
    return loss



class Dice(nn.Module):
    """ DiceLoss (optionally weighted)

        Args:
            * weights<ktensor|nparray|list|None>: crossentropy weights
        Returns:
            * weighted categorical crossentropy function
    """
    def __init__(self, weights=None):
        super(Dice, self).__init__()
        self.loss=weighted_categorical_crossentropy(weights)

        
    def forward(self, inpt, targ):
        return self.loss(inpt,targ)



