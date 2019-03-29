import torch.nn as nn
import torch.nn.functional as F
from pytorch_nns.models.shared.blocks import Conv as ConvBlock
from pytorch_nns.models.shared.blocks import Residule as ResBlock
from pytorch_nns.models.unet.blocks import Up as UNetUpBlock
import pytorch_nns.models.xception.blocks as blocks


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
            input_skip=True,
            dropout=False):
        super(XUnet,self).__init__()
        d1,d2,d3,d4,d5,d6,d7=self._get_dropouts(dropout)
        self.in1=ConvBlock(in_ch,32,stride=2,padding=1,depth=1)        
        self.in2=ConvBlock(32,64,padding=1,depth=1)
        self.d1=blocks.XDown(2,64,128,act_in=False,padding=1,dropout=d1)
        self.d2=blocks.XDown(2,128,256,padding=1,dropout=d2)
        self.d3=blocks.XDown(2,256,728,padding=1,dropout=d3)
        self.m=ResBlock(
            block=blocks.SeparableBlock(3,728,padding=1,dropout=d4),
            multiplier=1.0,
            in_ch=728)
        self.u2=blocks.XUp(2,728,256,padding=1,dropout=d5)
        self.u1=blocks.XUp(2,256,128,padding=1,dropout=d6)
        self.u0=blocks.XUp(2,128,64,padding=1,dropout=d7)
        self.out2=ConvBlock(64,32,padding=1,depth=1)
        self.input_skip=input_skip
        if input_skip:
            self.input_conv=ConvBlock(in_ch,16,padding=1,depth=2)
            self.uin=UNetUpBlock(32,32,out_up_ch=16,padding=1,depth=1)
        self.out1=ConvBlock(32,32,padding=1,depth=1)
        self.out0=ConvBlock(32,out_ch,kernel_size=1)
        self.output_activation=self._get_output_activation(
            output_activation,
            out_ch)


    def forward(self,inpt):
        x=self.in1(inpt)
        skip0=self.in2(x)
        skip1=self.d1(skip0)
        skip2=self.d2(skip1)
        x=self.d3(skip2)
        x=self.m(x)
        x=self.u2(x,skip2)
        x=self.u1(x,skip1)
        x=self.u0(x,skip0)
        x=self.out2(x)
        if self.input_skip:
            skip=self.input_conv(inpt)
            x=self.uin(x,skip)
        x=self.out1(x)
        x=self.out0(x)
        if self.output_activation:
            x=self.output_activation(x)
        return x


    def _get_dropouts(self,dropout):
        if isinstance(dropout,list):
            return dropout
        else:
            return [dropout]*7


    def _get_output_activation(self,output_activation,out_ch):
        if isinstance(output_activation,str):
            if output_activation.lower()=='sigmoid':
                act=nn.Sigmoid()
            elif output_activation.lower()=='softmax':
                act=nn.Softmax(dim=1)
            else:
                raise ValueError('[ERROR] unet: {output_activation} not implemented via str')
        elif output_activation is None:
            if out_ch==1:
                act=nn.Sigmoid()
            else:
                act=nn.Softmax(dim=1)
        elif output_activation is False:
            act=False
        else:
            if callable(output_activation()):
                act=output_activation()
            else:
                act
        return act  
