import torch.nn as nn
import torch.nn.functional as F
from pytorch_nns.models.shared.blocks import Conv as ConvBlock
import pytorch_nns.models.unet.blocks as blocks


class UNet(nn.Module):
    r""" UNet

    The standard UNet with optional addition of Squeeze and Excitation Blocks, 
    configurable network depth and depth of conv-blocks.  Allows for both valid
    and same padding.

    Note: currently designed for square (H=W) inputs only 

    Args:
        network_depth (int <2>): Depth of Network 'U'
        conv_depth (int <2>): The number of convolutional layers 
        in_ch (int): Number of channels in input
        in_size (int): Size (=H=W) of input
        out_ch (int <None>): Number of channels in output (if None => in_ch)
        init_ch (int <None>): Number of channels added in first convolution
        padding (int <0>): Padding
        bn (bool <True>): Add batch norm layer
        se (bool <True>): Add Squeeze and Excitation Block
        act (str <'relu'>): Method name of activation function
        act_kwargs (dict <{}>): Kwargs for activation function
        
    Properties:
        out_ch <int>: Number of channels of the output
        out_size <int>: Size (=H=W) of input
        padding <int>: Padding for conv block

    Links:
        https://arxiv.org/abs/1505.04597

    """
    def __init__(self,
            network_depth=4,
            conv_depth=2,
            in_size=572,
            in_ch=1,
            out_ch=2,
            init_ch=64,
            padding=0,
            bn=False,
            se=True):
        super(UNet, self).__init__()
        self.network_depth=network_depth
        self.conv_depth=conv_depth
        self.out_ch=out_ch
        self.padding=padding
        self.input_conv=ConvBlock(
            in_ch=in_ch,
            in_size=in_size,
            out_ch=init_ch,
            depth=self.conv_depth,
            padding=padding,
            bn=False,
            se=False)
        down_layers=self._down_layers(
            self.input_conv.out_ch,
            self.input_conv.out_size,
            bn=bn,
            se=se)
        self.down_blocks=nn.ModuleList(down_layers)
        up_layers=self._up_layers(down_layers,bn=bn,se=se)
        self.up_blocks=nn.ModuleList(up_layers)
        self.out_size=self.up_blocks[-1].out_size
        self.output_conv=self._output_layer(out_ch)

        
    def forward(self, x):
        x=self.input_conv(x)
        skips=[x]
        for block in self.down_blocks:
            x=block(x)
            skips.append(x)
        skips.pop()
        skips=skips[::-1]
        for skip,block in zip(skips,self.up_blocks):
            x=block(x,skip)
        x=self.output_conv(x)
        return F.softmax(x,dim=1)
    
    
    #
    # Internal Methods
    #
    def _down_layers(self,in_ch,in_size,bn,se):
        layers=[]
        for index in range(1,self.network_depth+1):
            layer=blocks.Down(
                in_ch,
                in_size,
                depth=self.conv_depth,
                padding=self.padding,
                bn=bn,
                se=se)
            in_ch=layer.out_ch
            in_size=layer.out_size
            layers.append(layer)
        return layers

        
    def _up_layers(self,down_layers,bn,se):
        down_layers=down_layers[::-1]
        down_layers.append(self.input_conv)
        first=down_layers.pop(0)
        in_ch=first.out_ch
        in_size=first.out_size
        layers=[]
        for down_layer in down_layers:       
            crop=blocks.Up.cropping(down_layer.out_size,2*in_size)
            layer=blocks.Up(
                in_ch,
                in_size,
                depth=self.conv_depth,
                crop=crop,
                padding=self.padding,
                bn=bn,
                se=se)
            in_ch=layer.out_ch
            in_size=layer.out_size
            layers.append(layer)
        return layers

    
    def _output_layer(self,out_ch):
        return nn.Conv2d(
           in_channels=64,
           out_channels=out_ch,
           kernel_size=1,
           stride=1,
           padding=0)
    
