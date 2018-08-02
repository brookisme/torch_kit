import torch.nn as nn
import pytorch_nns.functional as F



class WeightedCategoricalCrossentropy(nn.Module):
    """ weighted_categorical_crossentropy
        
        mean reduction of weighted categorical crossentropy

        Args:
            * weights<tensor|nparray|list>: category weights
    """
    def __init__(self, weights):
        super(WeightedCategoricalCrossentropy, self).__init__()
        self.weights=F.to_tensor(weights)

        
    def forward(self, inpt, targ):
        return F.weighted_categorical_crossentropy(inpt,targ,self.weights)



class Dice(nn.Module):
    """ (optionally weighted) dice loss
        
        mean reduction of (weighted) dice

        Args:
            * weights<tensor|nparray|list|None>: optional category weights
    """
    def __init__(self, weights):
        super(Dice, self).__init__()
        self.weights=F.to_tensor(weights)

        
    def forward(self, inpt, targ):
        return F.dice(inpt,targ,self.weights)



