import re
import torch.nn as nn
#
# CONSTANTS
#
SOFTMAX='Softmax'
LOG_SOFTMAX='LogSoftmax'
SIGMOID='Sigmoid'




#
# MODELING
#
def same_padding(kernel_size,dilation=1):
    size=kernel_size+((kernel_size-1)*(dilation-1))
    return int((size-1)//2)



#
# ACTIVATIONS
#
def activation(act=None,**act_kwargs):
    if isinstance(act,str):
        act=to_camel(act)
        act=re.sub('elu','ELU',act)
        act=re.sub('Elu','ELU',act)
        act=re.sub('rELU','ReLU',act)
        act=re.sub('RELU','ReLU',act)
        act=getattr(nn,act)(**act_kwargs)
    elif callable(act()):
        act=act(**act_kwargs)
    return act 


def output_activation(act=None,out_ch=None,**act_kwargs):
    if isinstance(act,str):
        act=to_camel(act)
        if act in [SOFTMAX,LOG_SOFTMAX]:
            act_kwargs['dim']=act_kwargs.get('dim',1)
        act=getattr(nn,act)(**act_kwargs)
    elif act is None:
        if out_ch==1:
            act=nn.Sigmoid()
        else:
            act=nn.Softmax(dim=1)
    elif callable(act()):
        act=act(**act_kwargs)
    return act 




#
# UTILS
#
def to_camel(string):
    parts=string.split('_')
    return ''.join([p.title() for p in parts])


