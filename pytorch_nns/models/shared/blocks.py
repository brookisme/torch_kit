import torch.nn as nn
import torch.nn.functional as F


RES_MULTIPLIER=0.75
class ResBlock(nn.Module):
    r""" ResBlock

    Args:
        block: block or list of layers
        multiplier <float [RES_MULTIPLIER]>: ident multiplier
        crop: <int|bool> 
            if <int> cropping = crop
            elif True calculate cropping
            else no cropping
    Links:
        TODO: GOT IDEA FROM FASTAI SOMEWHERE

    """
    def __init__(self,
            block,
            in_ch,
            out_ch,
            multiplier=RES_MULTIPLIER,
            crop=True):
        super(ResBlock, self).__init__()
        self.block=self._process_block(block)
        self.in_ch=in_ch
        self.out_ch=out_ch
        self.multiplier=multiplier
        self.crop=crop
        if self.in_ch!=self.out_ch:        
            self.ident_conv=nn.Conv2d(
                        in_channels=self.in_ch,
                        out_channels=self.out_ch,
                        kernel_size=1)
        else:
            self.ident_conv=False

    
    def forward(self, x):
        block_out=self.block(x)
        if self.crop:
            x=self._crop(x,block_out)
        if self.ident_conv:
            x=self.ident_conv(x)
        return (self.multiplier*x) + block_out 


    def _process_block(self,block):
        if isinstance(block,list):
            return nn.Sequential(*block)
        else:
            return block


    def _crop(self,x,layers_out):
        if not isinstance(self.crop,int):
            # get cropping
            out_size=layers_out.size()[-1]
            x_size=x.size()[-1]
            self.crop=(x_size-out_size)//2
        return x[:,:,self.crop:-self.crop,self.crop:-self.crop]




class SqueezeExcitation(nn.Module):
    r""" Squeeze and Excitation Block

    Args:
        nb_ch (int): Number of Channels in input image
        
    Links:
        https://arxiv.org/abs/1709.01507

    """
    def __init__(self, nb_ch, reduction=16, warn=True):
        super(SqueezeExcitation, self).__init__()
        self.nb_ch=nb_ch
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.reduction_ch=nb_ch//reduction
        if self.reduction_ch:
            self.fc=nn.Sequential(
                    nn.Linear(nb_ch,self.reduction_ch),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.reduction_ch, nb_ch),
                    nn.Sigmoid())
        elif warn:
            print('[WARNING] SqueezeExcitation skipped. nb_ch < reduction')

        
    def forward(self, x):
        if self.reduction_ch:
            y = self.avg_pool(x).view(-1,self.nb_ch)
            y = self.fc(y).view(-1,self.nb_ch,1,1)
            return x * y
        else:
            return x




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
        kernel_size (int <3>): Kernel Size
        stride (int <1>): Stride
        padding (int|str <0>): int or same if padding='same' -> int((kernel_size-1)/2) 
        res (bool <True>): ConvBlock -> ResBlock(ConvBlock)
        res_multiplier (float <RES_MULTIPLIER>)
        bn (bool <True>): Add batch norm layer
        se (bool <True>): Add Squeeze and Excitation Block
        act (str <'relu'>): Method name of activation function after each Conv Layer
        act_kwargs (dict <{}>): Kwargs for activation function after each Conv Layer
        
    Properties:
        out_ch <int>: Number of channels of the output
        out_size <int>: Size (=H=W) of input

    """
    #
    # CONSTANTS
    #
    SAME='same'


    #
    # STATIC METHODS
    #
    @staticmethod
    def same_padding(kernel_size):
        r""" calculates same padding size
        """
        return (kernel_size-1)//2


    #
    # PUBLIC METHODS
    #
    def __init__(self,
            in_ch,
            in_size,
            out_ch=None,
            depth=2, 
            kernel_size=3, 
            stride=1, 
            padding=0, 
            in_block_relu=True,
            res=False,
            res_multiplier=RES_MULTIPLIER,
            bn=False,
            se=False,
            act='ReLU',
            act_kwargs={}):
        super(Conv, self).__init__()
        same_padding=Conv.same_padding(kernel_size)
        if padding==Conv.SAME:
            padding=same_padding
            self.cropping=False
            self.out_size=in_size
        else:
            self.cropping=depth*(same_padding-padding)
            self.out_size=in_size-2*self.cropping
        self.in_ch=in_ch
        self.out_ch=out_ch or 2*in_ch
        self._set_conv_layers(
            depth,
            kernel_size,
            stride,
            padding,
            res,
            res_multiplier,
            act,
            act_kwargs)
        self._set_processes_layers(bn,se,act,act_kwargs)

        
    def forward(self, x):
        x=self.conv_layers(x)
        if self.bn:
            x=self.bn(x)
        if self.se:
            x=self.se(x)
        return x


    #
    # INTERNAL METHODS
    #    
    def _set_conv_layers(self,
            depth,
            kernel_size,
            stride,
            padding,
            res,
            res_multiplier,
            act,
            act_kwargs):
        layers=[]
        for index in range(depth):
            if index==0:
                in_ch=self.in_ch
            else:
                in_ch=self.out_ch
            layers.append(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=self.out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding))
            if act:
                layers.append(self._act_layer(act)(**act_kwargs))
        if res:
            self.conv_layers=ResBlock(
                layers,
                multiplier=res_multiplier,
                crop=self.cropping,
                in_ch=self.in_ch,
                out_ch=self.out_ch)
        else:
            self.conv_layers=nn.Sequential(*layers)


    def _set_processes_layers(self,bn,se,act,act_kwargs):
        if bn:
            self.bn=nn.BatchNorm2d(self.out_ch)
        else:
            self.bn=False
        if se:
            self.se=SqueezeExcitation(self.out_ch)
        else:
            self.se=False

        
    def _act_layer(self,act):
        if isinstance(act,str):
            return getattr(nn,act)
        else:
            return act


