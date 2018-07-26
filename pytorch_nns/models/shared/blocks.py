import torch.nn as nn
import torch.nn.functional as F




class SqueezeExcitation(nn.Module):
    r""" Squeeze and Excitation Block

    Args:
        nb_ch (int): Number of Channels in input image
        
    Links:
        https://arxiv.org/abs/1709.01507

    """
    def __init__(self, nb_ch, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.nb_ch=nb_ch
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
                nn.Linear(nb_ch, nb_ch // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(nb_ch // reduction, nb_ch),
                nn.Sigmoid())

        
    def forward(self, x):
        y = self.avg_pool(x).view(-1,self.nb_ch)
        y = self.fc(y).view(-1,self.nb_ch,1,1)
        return x * y




class Conv(nn.Module):
    r""" Conv Block

    Block of Convolutional Layers followed by optional BatchNorm, SqueezeExcitation, 
    and activation.

    Note: currently designed for square (H=W) inputs only 

    Args:
        in_ch (int): Number of channels in input
        in_size (int): Size (=H=W) of input
        out_ch (int <None>): Number of channels in output (if None => in_ch)
        depth (int <2>): The number of convolutional layers 
        kernel_size (int <3>): Kernal Size
        stride (int <1>): Stride
        padding (int <0>): Padding
        bn (bool <True>): Add batch norm layer
        se (bool <True>): Add Squeeze and Excitation Block
        act (str <'relu'>): Method name of activation function
        act_kwargs (dict <{}>): Kwargs for activation function
        
    Properties:
        out_ch <int>: Number of channels of the output
        out_size <int>: Size (=H=W) of input

    """
    def __init__(self,
            in_ch,
            in_size,
            out_ch=None,
            depth=2, 
            kernel_size=3, 
            stride=1, 
            padding=0, 
            bn=True,
            se=True,
            act='relu',
            act_kwargs={}):
        super(Conv, self).__init__()
        self.out_ch=out_ch or 2*in_ch
        self._set_post_processes(self.out_ch,bn,se,act,act_kwargs)
        self._set_conv_layers(
            depth,
            in_ch,
            kernel_size,
            stride,
            padding)
        self.out_size=in_size-depth*2*((kernel_size-1)/2-padding)

        
    def forward(self, x):
        x=self.conv_layers(x)
        if self.bn:
            x=self.bn(x)
        if self.act:
            x=self._activation(x)
        if self.se:
            x=self.se(x)
        return x

    
    def _set_post_processes(self,out_channels,bn,se,act,act_kwargs):
        if bn:
            self.bn=nn.BatchNorm2d(out_channels)
        else:
            self.bn=False
        if se:
            self.se=SqueezeExcitation(out_channels)
        else:
            self.se=False
        self.act=act
        self.act_kwargs=act_kwargs

        
    def _set_conv_layers(
            self,
            depth,
            in_ch,
            kernel_size,
            stride,
            padding):
        layers=[]
        for index in range(depth):
            if index!=0:
                in_ch=self.out_ch
            layers.append(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=self.out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding))
        self.conv_layers=nn.Sequential(*layers)

        
    def _activation(self,x):
        return getattr(F,self.act,**self.act_kwargs)(x)
