import torch.nn as nn
import pytorch_nns.helpers as h
import pytorch_nns.functional as f



class WeightedCategoricalCrossentropy(nn.Module):
    """ weighted_categorical_crossentropy
        
        mean reduction of weighted categorical crossentropy

        Args:
            * weights<tensor|nparray|list>: category weights
            * device<str|None>: device-name. if exists, send weights to specified device
    """
    def __init__(self, weights, device=None):
        super(WeightedCategoricalCrossentropy, self).__init__()
        self.weights=h.to_tensor(weights)
        if device:
            self.weights=self.weights.to(device)

    def forward(self, inpt, targ):
        return f.weighted_categorical_crossentropy(inpt,targ,self.weights)



class Dice(nn.Module):
    """ (optionally weighted) dice loss
        
        mean reduction of (weighted) dice

        Args:
            * weights<tensor|nparray|list|None>: optional category weights
            * device<str|None>: device-name. if exists, send weights to specified device
    """
    def __init__(self, weights=None, device=None):
        super(Dice, self).__init__()
        if weights:
            weights=h.to_tensor(weights)
            if device:
                weights=weights.to(device)
        self.weights=weights
        
    def forward(self, inpt, targ):
        # return f.dice(inpt,targ,self.weights)
        return f.soft_dice_loss(inpt,targ,self.weights)



