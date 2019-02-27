import torch.nn as nn
import torch.nn.functional as F
from pytorch_nns.models.shared.blocks import Conv as ConvBlock
from pytorch_nns.models.shared.blocks import Residule as ResBlock
from pytorch_nns.models.unet.blocks import Up as UNetUpBlock
import pytorch_nns.models.xception.blocks as blocks


class XUnet(nn.Module):
    r""" XUnet
    
    ** hardcoded first pass **
    
    Args:
    Properties:
    Links:

    """
    def __init__(self,input_stride=2):
        super(XUnet,self).__init__()
        self.in1=ConvBlock(8,32,stride=input_stride,padding=1,depth=1)        
        self.in2=ConvBlock(32,64,padding=1,depth=1)
        self.d1=blocks.XDown(2,64,128,act_in=False,padding=1)
        self.d2=blocks.XDown(2,128,256,padding=1)
        self.d3=blocks.XDown(2,256,728,padding=1)
        self.m=ResBlock(
            block=blocks.SeparableBlock(3,728,padding=1),
            multiplier=1.0,
            in_ch=728)
        self.u2=blocks.XUp(2,728,256,padding=1)
        self.u1=blocks.XUp(2,256,128,padding=1)
        self.u0=blocks.XUp(2,128,64,padding=1)
        self.out2=ConvBlock(64,32,padding=1,depth=1)
        if input_stride==2:
            self.input_conv=ConvBlock(8,16,padding=1,depth=2)
            self.uin=UNetUpBlock(32,32,out_up_ch=16,padding=1,depth=1)
        else:
            self.uin=False
        self.out1=ConvBlock(32,32,padding=1,depth=1)
        self.out0=ConvBlock(32,5,kernel_size=1)
        
        
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
        if self.uin:
            skip=self.input_conv(inpt)
            x=self.uin(x,skip)
        x=self.out1(x)
        x=self.out0(x)
        return x

