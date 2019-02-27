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
            down_block=blocks.Down,
            up_block=blocks.Up,
            input_conv=ConvBlock,
            output_conv=ConvBlock,
            output_activation=None,
            default_config={},
            input_config=None,
            down_config=None,
            up_config=None,
            output_config=None):
        super(UNet, self).__init__()
        self.default_config=default_config
        if input_conv:
            self.input_conv=input_conv(
                in_ch,
                input_out_ch,
                **self._config(input_config))
        self.down_blocks, out_ch=self._down_layers_and_out_ch(
            input_out_ch,depth,
            down_block,
            self._config(down_config))
        if up_block:
            self.up_blocks, out_ch=self._up_layers_and_out_ch(
                out_ch,depth,
                up_block,
                self._config(up_config))
        else:
            self.up_blocks=None
        if output_conv:
            self.output_conv=output_conv(
                out_ch,
                output_channels,
                **self._config(output_config))
        self.output_activation=output_activation


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
    
    
    def _config(self,config):
        if config is None:
            return self.default_config
        else:
            return config


