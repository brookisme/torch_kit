import torch
import torch.nn as nn
import torch.nn.functional as F
import models.unet.blocks as blocks


class UNet(nn.Module):

    def __init__(self,
            network_depth=4,
            conv_depth=2,
            in_size=572,
            in_ch=1,
            out_ch=2,
            init_ch=64,
            padding=0,
            bn=True,
            se=True,
            act='relu',
            act_kwargs={}):
        super(UNet, self).__init__()
        self.network_depth=network_depth
        self.conv_depth=conv_depth
        self.out_ch=out_ch
        self.padding=padding
        self.input_conv=blocks.Conv(
            in_ch=in_ch,
            in_size=in_size,
            out_ch=init_ch,
            depth=self.conv_depth,
            padding=padding,
            bn=bn,
            se=se,
            act=act,
            act_kwargs=act_kwargs)
        down_layers=self._down_layers(
            self.input_conv.out_ch,
            self.input_conv.out_size,
            bn=bn,
            se=se,
            act=act,
            act_kwargs=act_kwargs)
        self.down_blocks=nn.ModuleList(down_layers)
        up_layers=self._up_layers(
            down_layers,
            bn=bn,
            se=se,
            act=act,
            act_kwargs=act_kwargs)
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
        return x
    
    
    def _down_layers(self,in_ch,in_size,bn,se,act,act_kwargs):
        layers=[]
        for index in range(1,self.network_depth+1):
            layer=blocks.Down(
                in_ch,
                in_size,
                depth=self.conv_depth,
                padding=self.padding,
                bn=bn,
                se=se,
                act=act,
                act_kwargs=act_kwargs)
            in_ch=layer.out_ch
            in_size=layer.out_size
            layers.append(layer)
        return layers

        
    def _up_layers(self,down_layers,bn,se,act,act_kwargs):
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
                se=se,
                act=act,
                act_kwargs=act_kwargs)
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
    
