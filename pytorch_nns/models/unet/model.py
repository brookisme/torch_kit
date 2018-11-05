import torch.nn as nn
import torch.nn.functional as F
from pytorch_nns.models.shared.blocks import Conv as ConvBlock, RES_MULTIPLIER
import pytorch_nns.models.unet.blocks as blocks


class UNet(nn.Module):
    r""" UNet

    The standard UNet with optional addition of Squeeze and Excitation Blocks, 
    configurable network depth and depth of conv-blocks.  Allows for both valid
    and same padding.

    Note: currently designed for square (H=W) inputs only 

    Args:
        network_depth (int <2>): Depth of Network 'U' 
        kernel_sizes (list <None>): Overides network depth and kernel size
        conv_depth (int <2>): The number of convolutional layers 
        in_ch (int): Number of channels in input
        in_size (int): Size (=H=W) of input
        out_ch (int <None>): Number of channels in output (if None => in_ch)
        input_out_ch (int <None>): Number of out channels in first conv_block
        output_in_ch (int <None>): Number of in channels in last conv_block
        padding (int <0>): Padding
        res (bool <False>): ConvBlock -> ResBlock(ConvBlock)
        res_up (bool|None <False>): 
            ConvBlock -> ResBlock(ConvBlock) for Up path
            if None = res
        res_multiplier (float <blocks.RES_MULTIPLIER>)
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
            kernel_sizes=None,
            conv_depth=2,
            in_size=572,
            in_ch=1,
            out_ch=2,
            input_out_ch=64,
            output_in_ch=64,
            padding=0,
            kernel_size=3,
            res=False,
            res_up=None,
            res_multiplier=RES_MULTIPLIER,            
            bn=False,
            se=True,
            se_up=None,
            output_activation=None,
            act='ReLU',
            act_kwargs={}):
        super(UNet, self).__init__()
        self.network_depth=network_depth
        self.conv_depth=conv_depth
        self.out_ch=out_ch
        self.padding=padding
        self.kernel_sizes=kernel_sizes or [kernel_size]*network_depth
        self.input_conv=ConvBlock(
            in_ch=in_ch,
            in_size=in_size,
            out_ch=input_out_ch,
            padding=padding,
            kernel_size=kernel_size,
            depth=self.conv_depth,
            res=res,
            res_multiplier=res_multiplier,
            bn=False,
            se=False,
            act=act,
            act_kwargs=act_kwargs)
        down_layers=self._down_layers(
            self.input_conv.out_ch,
            self.input_conv.out_size,
            res=res,
            res_multiplier=res_multiplier,
            bn=bn,
            se=se,
            act=act,
            act_kwargs=act_kwargs)
        self.down_blocks=nn.ModuleList(down_layers)
        if res_up is None:
            res_up=res
        if se_up is None:
            se_up=se
        up_layers=self._up_layers(
            down_layers,
            res=res_up,
            res_multiplier=res_multiplier,
            bn=bn,
            se=se_up,
            act=act,
            act_kwargs=act_kwargs)
        self.up_blocks=nn.ModuleList(up_layers)
        self.out_size=self.up_blocks[-1].out_size
        self.output_conv=self._output_layer(output_in_ch,out_ch)
        self.output_activation=self._get_output_activation(output_activation)

        
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
        if self.output_activation:
            x=self.output_activation(x)
        return x
    
    
    #
    # Internal Methods
    #
    def _down_layers(self,
            in_ch,
            in_size,
            res,
            res_multiplier,
            bn,
            se,
            act,
            act_kwargs):
        layers=[]
        for kernel_size in self.kernel_sizes:
            layer=blocks.Down(
                in_ch,
                in_size,
                depth=self.conv_depth,
                padding=self.padding,
                kernel_size=kernel_size,
                res=res,
                res_multiplier=res_multiplier,
                bn=bn,
                se=se,
                act=act,
                act_kwargs=act_kwargs)
            in_ch=layer.out_ch
            in_size=layer.out_size
            layers.append(layer)
        return layers

        
    def _up_layers(
            self,
            down_layers,
            res,
            res_multiplier,
            bn,
            se,
            act,
            act_kwargs):
        down_layers=down_layers[::-1]
        down_layers.append(self.input_conv)
        first=down_layers.pop(0)
        in_ch=first.out_ch
        in_size=first.out_size
        layers=[]
        up_ks=self.kernel_sizes[::-1]
        for down_layer,kernel_size in zip(down_layers,up_ks):
            crop=blocks.Up.cropping(down_layer.out_size,2*in_size)
            layer=blocks.Up(
                in_ch,
                in_size,
                depth=self.conv_depth,
                crop=crop,
                padding=self.padding,
                kernel_size=kernel_size,
                res=res,
                res_multiplier=res_multiplier,
                bn=bn,
                se=se,
                act=act,
                act_kwargs=act_kwargs)
            in_ch=layer.out_ch
            in_size=layer.out_size
            layers.append(layer)
        return layers

    
    def _output_layer(self,in_ch,out_ch):
        return nn.Conv2d(
           in_channels=in_ch,
           out_channels=out_ch,
           kernel_size=1,
           stride=1,
           padding=0)


    def _get_output_activation(self,output_activation):
        if isinstance(output_activation,str):
            if output_activation.lower()=='sigmoid':
                act=nn.Sigmoid()
            elif output_activation.lower()=='softmax':
                act=nn.Softmax(dim=1)
            else:
                raise ValueError('[ERROR] unet: {output_activation} not implemented via str')
        elif output_activation is None:
            if self.out_ch==1:
                act=nn.Sigmoid()
            else:
                act=nn.Softmax(dim=1)
        elif output_activation is False:
            act=False
        else:
            act=output_activation()
        return act
    
