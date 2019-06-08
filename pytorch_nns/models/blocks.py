import torch.nn as nn
import torch.nn.functional as F
from pytorch_nns.models.helpers import activation, same_padding




#
# BLOCKS
#
class Conv(nn.Module):
    r""" Conv Block
    
    Args:
        in_ch (int): Number of channels in input
        out_ch (int <None>): Number of channels in output (if None => in_ch)
        out_chs: LIST
        depth (int <2>): The number of convolutional layers 
        kernel_size (int <3>): Kernel Size
        kernel_sizes: LIST
        stride (int <1>): Stride
        dilation (int <1>): Dilation rate
        padding (int|str <0>): int or same if padding='same' -> int((kernel_size-1)/2) 
        batch_norm (bool <True>): Add batch norm after conv
        dropout (False|float <False>): Dropout to be applied after Conv
        act (str <'relu'>): Method name of activation function after each Conv Layer
        act_config (dict <{}>): Kwargs for activation function after each Conv Layer
    """
    #
    # CONSTANTS
    #
    SAME='same'
    RELU='ReLU'


    #
    # PUBLIC METHODS
    #
    def __init__(self,
            in_ch,
            out_ch=None,
            out_chs=None,
            depth=1, 
            kernel_size=3, 
            kernel_sizes=None,
            stride=1,
            dilation=1,
            padding=SAME, 
            batch_norm=True,
            dropout=False,
            act=RELU,
            act_config={}):
        super(Conv, self).__init__()
        self.in_ch=int(in_ch)
        if out_ch is None:
            out_ch=self.in_ch
        else:
            out_ch=int(out_ch)
        if out_chs is None:
            out_chs=[out_ch]*depth
        self.out_ch=out_chs[-1]
        if kernel_sizes is None:
            kernel_sizes=[kernel_size]*depth
        self.padding=padding
        self.conv_blocks=self._conv_blocks(
            in_ch,
            out_chs,
            kernel_sizes,
            stride,
            dilation,
            batch_norm,
            dropout,
            act,
            act_config)

        
    def forward(self, x):
        return self.conv_blocks(x)


    #
    # INTERNAL METHODS
    #    
    def _conv_blocks(
            self,
            in_ch,
            out_chs,
            kernel_sizes,
            stride,
            dilation,
            batch_norm,
            dropout,
            act,
            act_config):
        layers=[]
        for ch,k in zip(out_chs,kernel_sizes):
            layers.append(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=ch,
                    kernel_size=k,
                    stride=stride,
                    padding=self._padding(k,dilation),
                    dilation=dilation,
                    bias=(not batch_norm)))
            if batch_norm:
                layers.append(nn.BatchNorm2d(self.out_ch))
            if act:
                layers.append(activation(act=act,**act_config))
            if dropout:
                layers.append(nn.Dropout2d(p=dropout))
            in_ch=ch
        return nn.Sequential(*layers)


    def _padding(self,kernel_size,dilation):
        if self.padding==Conv.SAME:
            return same_padding(kernel_size,dilation)
        else:
            return self.padding




class Dense(nn.Module):
    r""" Dense Block
    
    Args:
        in_ch (int): Number of channels in input
        out_ch (int <None>): Number of channels in output (if None => in_ch)
        out_chs: LIST
        depth (int <2>): The number of convolutional layers 
        batch_norm (bool <True>): Add batch norm after conv
        dropout (False|float <False>): Dropout to be applied after Conv
        act (str <'relu'>): Method name of activation function after each Conv Layer
        act_config (dict <{}>): Kwargs for activation function after each Conv Layer
    """
    #
    # PUBLIC METHODS
    #
    def __init__(self,
            in_ch,
            out_ch=None,
            out_chs=None,
            depth=1, 
            batch_norm=True,
            dropout=False,
            act='ReLU',
            act_config={}):
        super(Dense, self).__init__()
        self.in_ch=int(in_ch)
        if out_ch is None:
            out_ch=self.in_ch
        else:
            out_ch=int(out_ch)
        if out_chs is None:
            out_chs=[out_ch]*depth
        self.out_ch=out_chs[-1]
        self.dense_blocks=self._dense_blocks(
            self.in_ch,
            out_chs,
            batch_norm,
            dropout,
            act,
            act_config)

        
    def forward(self, x):
        return self.dense_blocks(x)


    #
    # INTERNAL METHODS
    #    
    def _dense_blocks(
            self,
            in_ch,
            out_chs,
            batch_norm,
            dropout,
            act,
            act_config):
        layers=[]
        for ch in out_chs:
            layers.append(
                nn.Linear(
                    in_features=in_ch,
                    out_features=ch,
                    bias=(not batch_norm)))
            if batch_norm:
                layers.append(nn.BatchNorm1d(self.out_ch))
            if act:
                layers.append(activation(act=act,**act_config))
            if dropout:
                layers.append(nn.Dropout2d(p=dropout))
            in_ch=ch
        return nn.Sequential(*layers)


