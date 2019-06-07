import torch.nn as nn
import torch.nn.functional as F
import pytorch_nns.helpers as h
import pytorch_nns.models.deeplab.blocks as blocks
from pytorch_nns.models.xception.model import Xception



class DeeplabV3plus(nn.Module):
    r""" DeeplabV3+

    Properties:
    Links:

    """
    XCEPTION='xception'


    def __init__(self,backbone=XCEPTION):
        super(DeeplabV3plus,self).__init__()
        self.backbone=self._backbone(backbone)
        self.aspp=blocks.ASPP('...args...')


    def forward(self,x):
        x,lowx=self.backbone(x)
        x=self.aspp(x)
        x=self.up1(x)
        x=torch.cat([x,lowx])
        x=self.low_level_convs(x)
        x=self.up2(x)
        return x


    def _backbone(self,backbone):
        if backbone==DeeplabV3plus.XCEPTION:
            return Xception('...args...')
        else:
            raise NotImplementedError("Currently only supports 'xception' backbone")