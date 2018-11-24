import random
import torch
import torch.nn as nn


class Scale(nn.Module):
    
    def __init__(self,init_value=1.0,delta=0,device=None):
        """ Scale
        
        pytorch-layer with single parameter to rescale input 

        Args:
            * init_value<int>: initial value of weight (default=1.0)
        """
        super(Scale, self).__init__()
        self.init_value=init_value
        self.device=device
        self.delta=delta
        self.weight=nn.Parameter(data=torch.Tensor(1), requires_grad=True)
        self.reset_weight()


    def reset_weight(self):
        w=self.init_value+random.uniform(-self.delta,self.delta)
        self.weight.data=torch.Tensor([w]).to(self.device)


    def forward(self, x):
        return self.weight*x