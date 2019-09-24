import torch.nn as nn
import torch_kit.helpers as h
import torch_kit.functional as f



class MaskedLoss(nn.Module):
    """ masked_loss

        negative-log-likelyhood or cross-entropy loss with mask

        Args:
            * mask_value<int|array|None>: keep values where target!=mask
            * loss_type<str['nll']>: nll or ce
            * device<str|None>: device-name. if exists, send weights to specified device
    """
    def __init__(self,mask_value=None,loss_type='nll',device=None):
        super(MaskedLoss, self).__init__()
        if loss_type=='nll':
            self.loss_layer=nn.NLLLoss(reduction='none')
        else:
            self.loss_layer=nn.CrossEntropyLoss(reduction='none')
        self.mask_value=mask_value


    def forward(self, inpt, targ):
        loss=self.loss_layer(inpt,targ)
        if self.mask_value is not None:
            mask=(targ!=self.mask_value)
            loss=loss[mask]
        return loss.mean()
        

class WeightedCategoricalCrossentropy(nn.Module):
    """ weighted_categorical_crossentropy
        
        mean reduction of weighted categorical crossentropy

        Args:
            * weights<tensor|nparray|list>: category weights
            * device<str|None>: device-name. if exists, send weights to specified device
    """
    def __init__(self, weights=None,nb_categories=None,device=None):
        super(WeightedCategoricalCrossentropy, self).__init__()
        if weights is None:
            weights=[1.0]*nb_categories
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


