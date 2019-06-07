import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_nns.helpers as h
import pytorch_nns.models.deeplab.blocks as blocks
from pytorch_nns.models.xception.model import Xception



class DeeplabV3plus(nn.Module):
    r""" DeeplabV3+

    Args:
        backbone<str>: 'xception' is the only implemented backbone
        aspp_ch<int>: 
            the number of output channels of aspp atrous blocks and 
            the aspp block itself. defaults to 256
    Properties:
    Links:

    """
    XCEPTION='xception'


    def __init__(self,
            in_ch,
            out_ch,
            backbone_config={},
            backbone_low_level_out_ch=None,
            backbone_out_ch=None,
            backbone=XCEPTION,
            aspp_out_ch=256,
            upsample_mode='bilinear'):
        super(DeeplabV3plus,self).__init__()
        self.backbone,backbone_out_ch,backbone_low_level_out_ch=self._backbone(
            backbone,
            in_ch,
            backbone_config,
            backbone_out_ch,
            backbone_low_level_out_ch)
        self.aspp=blocks.ASPP(in_ch=backbone_out_ch,out_ch=aspp_out_ch)
        self.channel_reducer=nn.Conv2d(
            in_channels=aspp_out_ch+backbone_low_level_out_ch,
            out_channels=out_ch,
            kernel_size=1)
        self.up1=nn.Upsample(scale_factor=4,mode=upsample_mode,align_corners=False)
        self.up2=nn.Upsample(scale_factor=4,mode=upsample_mode,align_corners=False)


    def forward(self,x):
        x,lowx=self.backbone(x)
        x=self.aspp(x)
        x=self.up1(x)
        x=torch.cat([x,lowx],dim=1)
        x=self.channel_reducer(x)
        x=self.up2(x)
        return x


    def _backbone(self,backbone,in_ch,backbone_config,out_ch,low_level_out_ch):
        if backbone==DeeplabV3plus.XCEPTION:
            net=Xception(in_ch=in_ch,**backbone_config)
            out_ch=out_ch or net.out_ch
            low_level_out_ch=low_level_out_ch or net.low_level_out_ch
            return net, out_ch, low_level_out_ch
        else:
            raise NotImplementedError("Currently only supports 'xception' backbone")





