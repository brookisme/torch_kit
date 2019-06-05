import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch=None,
            kernel_size=3,
            stride=1,
            bias=False,
            batch_norm=True,
            act='ReLU',
            act_config={}):
        super(SeparableConv2d, self).__init__()
        pass



class EntryBlock(nn.Module):
    """ Xception Entry Block """
    def __init__(
            self,
            in_ch,
            entry_ch=32,
            entry_out_ch=64
        ):
        super(EntryBlock, self).__init__()
        self.sconv=SeparableConv2d(
            in_ch=in_ch,
            out_ch=entry_ch,
            stride=2)
        self.conv=Conv(
            in_ch=entry_ch
            out_ch=entry_out_ch)


    def forward(self,x):
        x=self.sconv(x)
        return self.conv(x)



class  XBlock(object):
    """ Xception Block """
    def __init__(self,
            in_ch,
            out_ch=None,
            depth=3
        ):
        super(XBlock, self).__init__()
        if out_ch is None:
            out_ch=in_ch
        self.sconv_in=SeparableConv2d(in_ch,out_ch)
        self.sconv_blocks_depth=depth-2
        if sconv_blocks_depth:
            self.sconv_blocks=self._sconv_blocks(out_ch)
        self.sconv_strided=SeparableConv2d(
                in_ch=out_ch,
                out_ch=out_ch,
                stride=2)
        self.pointwise_conv=Conv(
            in_ch=in_ch
            out_ch=out_ch,
            stride=2,
            kernel_size=1)


    def forward(self,x):
        xpc=self.pointwise_conv(x)
        x=self.sconv_in(x)
        if sconv_blocks_depth:
            x=self.sconv_blocks(x)
        x=self.sconv_strided(x)
        return x.add_(xpc)


    def _sconv_blocks(self,ch):
        blocks=[]
        for _ in range(self.sconv_blocks_depth):
            blocks.append(SeparableConv2d(ch))
        return nn.ModuleList(blocks)


class SeparableResStack(object):
    """  Xception Middle Flow """
    def __init__(self,
            ch,
            depth=16,
        ):
        super(SeparableResStack, self).__init__()
        self.sconvs=self._sconv_blocks(ch,depth)


    def forward(self,x):
        return x.add_(sconvs(x))


    def _sconv_blocks(self,ch,depth):
        blocks=[]
        for _ in range(depth):
            blocks.append(SeparableConv2d(ch))
        return nn.ModuleList(blocks)



class SeparableStack(object)
    """ TODO: 
    * CHECK THIS
    * USE FOR EXITSTACK/SeparableResStack/XBLOCK 
    """
    def __init__(self,in_ch,out_ch=None,depth=1,out_chs=None):
        super(SeparableStack, self).__init__()
        if out_ch:
            out_chs=[out_ch]*depth
        self.sconvs=self._sconv_blocks(in_ch,out_chs)


    def forward(self,x):
        return x.add_(sconvs(x))


    def _sconv_blocks(self,ch,depth):
        blocks=[]
        for ch in range(out_chs):
            blocks.append(SeparableConv2d(in_ch,ch))
            in_ch=ch
        return nn.ModuleList(blocks)






