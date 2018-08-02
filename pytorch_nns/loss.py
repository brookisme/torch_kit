import torch.nn as nn
import pytorch_nns.helpers as h
import pytorch_nns.functional as f



class WeightedCategoricalCrossentropy(nn.Module):
    """ weighted_categorical_crossentropy
        
        mean reduction of weighted categorical crossentropy

        Args:
            * weights<tensor|nparray|list>: category weights
    """
    def __init__(self, weights, force_cpu=False):
        super(WeightedCategoricalCrossentropy, self).__init__()
        self.weights=h.to_tensor(weights)
        print('FCPU',h.get_device(force_cpu))
        self.weights.to(h.get_device(force_cpu))
        
    def forward(self, inpt, targ):
        return f.weighted_categorical_crossentropy(inpt,targ,self.weights)



class Dice(nn.Module):
    """ (optionally weighted) dice loss
        
        mean reduction of (weighted) dice

        Args:
            * weights<tensor|nparray|list|None>: optional category weights
    """
    def __init__(self, weights, force_cpu=False):
        super(Dice, self).__init__()
        self.weights=h.to_tensor(weights)
        self.weights.to(h.get_device(force_cpu))

        
    def forward(self, inpt, targ):
        return f.dice(inpt,targ,self.weights)



