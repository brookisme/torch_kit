import torch.nn as nn
import torch.nn.functional as F
import pytorch_nns.helpers as h
from pytorch_nns.models.shared.blocks import Conv as ConvBlock
from pytorch_nns.models.shared.blocks import Residule as ResBlock
from pytorch_nns.models.unet.blocks import Up as UNetUpBlock
import pytorch_nns.models.xunet.blocks as blocks



class XUnet(nn.Module):
    r""" XUnet
    
    ** first pass **
    
    Args:
    Properties:
    Links:

    """
    def __init__(self,
            in_ch,
            out_ch,
            output_activation=None,
            output_type=None,
            padding=1,
            input_skip=True,
            entry_conv_ch=64,
            down_channels=[128,256,512],
            bottleneck_depth=3,
            dropouts={}):
        super(XUnet,self).__init__()
        # entry
        self.entry_conv=self._entry_conv(
            in_ch,
            entry_conv_ch,
            padding=1,
            dropout=dropouts.get('entry'))
        # unet
        self.down_blocks=self._down_blocks(
            entry_conv_ch,
            down_channels,
            padding=padding,
            dropout=dropouts.get('down'))
        self.bottleneck=self._bottleneck(
            bottleneck_depth,
            down_channels[-1],
            padding=padding,
            dropout=dropouts.get('bottleneck'))
        self.up_blocks=self._up_blocks(
            down_channels[::-1],
            entry_conv_ch,
            padding=padding,
            dropout=dropouts.get('up'))
        # exit
        self.exit1=ConvBlock(
            entry_conv_ch,
            entry_conv_ch//2,
            depth=1,
            padding=padding,
            dropout=dropouts.get('exit1'))
        self.input_skip=input_skip
        if self.input_skip:
            self.input_conv=ConvBlock(
                in_ch,
                entry_conv_ch//4,
                padding=padding,
                depth=2,
                dropout=dropouts.get('input_conv'))
            cb_in_ch=entry_conv_ch//2
        else:
            cb_in_ch=entry_conv_ch//4
        self.exit2=UNetUpBlock(
            entry_conv_ch//2,
            cb_in_ch=cb_in_ch,
            padding=padding,
            depth=1,
            dropout=dropouts.get('exit2'))
        # out
        self.output_type=output_type
        self.out=ConvBlock(
            entry_conv_ch//4,
            out_ch,
            kernel_size=1,
            dropout=dropouts.get('out'))
        self.output_activation=h.get_output_activation(
            output_activation,
            out_ch)



    def forward(self,inpt):
        inpt=inpt.float()
        skips=[self.entry_conv(inpt)]
        for block in self.down_blocks:
            skips.append(block(skips[-1]))
        x=self.bottleneck(skips.pop())
        skips.reverse()
        for skip,block in zip(skips,self.up_blocks):
            x=block(x,skip)
        x=self.exit1(x)
        if self.input_skip:
            skip=self.input_conv(inpt)
        else:
            skip=None
        x=self.exit2(x,skip)
        x=self.out(x)
        x=self._require_type(x)
        if self.output_activation:
            x=self.output_activation(x)
        return x


    def _entry_conv(self,in_ch,entry_conv_ch,padding,dropout):
        c1=ConvBlock(
            in_ch,
            entry_conv_ch//2,
            stride=2,
            padding=padding,
            depth=1,
            dropout=dropout)
        c2=ConvBlock(
            entry_conv_ch//2,
            entry_conv_ch,
            padding=padding,
            depth=1,
            dropout=dropout)
        return nn.Sequential(c1,c2)


    def _down_blocks(self,in_ch,down_channels,padding,dropout=False):
        act_in=False
        layers=[]
        if not isinstance(dropout,list):
            dropout=[dropout]*len(down_channels)
        for ch,drp in zip(down_channels,dropout):
            blk=blocks.XDown(2,in_ch,ch,act_in=act_in,padding=padding,dropout=drp)
            layers.append(blk)
            in_ch=ch
            act_in=True
        return nn.ModuleList(layers)


    def _bottleneck(self,depth,in_ch,padding,dropout=False):
            return ResBlock(
                block=blocks.SeparableBlock(
                    depth=depth,
                    in_ch=in_ch,
                    padding=padding,
                    dropout=dropout),
                multiplier=1.0,
                in_ch=in_ch,
                crop=(padding==0))


    def  _up_blocks(self,up_channels,out_ch,padding,dropout=False):
        up_channels.append(out_ch)
        layers=[]
        nb_layers=len(up_channels)-1
        if not isinstance(dropout,list):
            dropout=[dropout]*nb_layers
        for i in range(nb_layers):
            in_ch=up_channels[i]
            out_ch=up_channels[i+1]
            layers.append(blocks.XUp(2,in_ch,out_ch,padding=padding,dropout=dropout[i]))
        return nn.ModuleList(layers)


    def _require_type(self,x):
        if self.output_type=='float':
            x=x.float()
        elif self.output_type=='double':
            x=x.double()        
        elif self.output_type=='long':
            x=x.long()
        return x


 
