import torch.nn as nn
import torch.nn.functional as F
from pytorch_nns.models.shared.blocks import Conv as ConvBlock, RES_MULTIPLIER
from pytorch_nns.models.shared.blocks import GeneralizedConvResnet
import pytorch_nns.models.unet.blocks as blocks


class UNet(nn.Module):
    r""" UNet

    Args:
        
    Properties:

    Links:
        https://arxiv.org/abs/1505.04597

    """    
    def __init__(self,
            in_ch,
            input_out_ch,
            output_channels,
            depth,
            down_block,
            up_block,
            input_conv=None,
            bottleneck_block=None,
            output_conv=None,
            output_activation=None,
            conv_config={},
            input_config={},
            down_config={},
            up_config={},
            output_config={}):
        super(UNet, self).__init__()
        if not output_conv:
            output_conv=input_conv
        self.input_conv=input_conv(in_ch,input_out_ch,**(input_config or conv_config)) # could pass
        self.down_blocks, out_ch=self._down_layers_and_out_ch(input_out_ch,depth,down_block,down_config or conv_config)
        if up_block:
            self.up_blocks, out_ch=self._up_layers_and_out_ch(out_ch,depth,up_block,up_config or conv_config)
        else:
            self.up_blocks=None
        self.output_conv=output_conv(out_ch,output_channels,**(output_config or conv_config))  # could pass
        self.output_activation=output_activation
        
        
    def _down_layers_and_out_ch(self,in_ch,depth,down_block,down_config):
        layers=[]
        for _ in range(depth):
            layer=down_block(in_ch=in_ch,out_ch=in_ch*2,**down_config)
            layers.append(layer)
            in_ch*=2
        return nn.ModuleList(layers), in_ch
    
    
    def _up_layers_and_out_ch(self,in_ch,depth,up_block,up_config):
        layers=[]
        for _ in range(depth):
            layer=up_block(in_ch=in_ch,out_ch=in_ch//2,**up_config)
            layers.append(layer)
            in_ch/=2
        return nn.ModuleList(layers), in_ch
    
    
    def forward(self, x):
        x=self.input_conv(x)
        skips=[x]
        for block in self.down_blocks:
            x=block(x)
            skips.append(x)
        if self.up_blocks:
            skips.pop()
            skips=skips[::-1]
            for skip,block in zip(skips,self.up_blocks):
                x=block(x,skip)
        else:
            del(skips)
        x=self.output_conv(x)
        if self.output_activation:
            x=self.output_activation(x)
        return x
