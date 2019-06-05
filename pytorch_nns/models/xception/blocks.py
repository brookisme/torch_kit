import torch
import torch.nn as nn



#
# GENERAL BLOCKS
#
class SeparableConv2d(nn.Module):
    """ stack of SeparableConv2d
    Args:
        in_ch<int>: number of input channels
        out_ch<int|None>: if out_ch is None out_ch=in_ch
        kernel_size<int>: kernel_size
        stride<int>: stride
        bias<bool|None>: 
            - include bias term.  
            - if None bias=(not batch_norm)
        act<str|func>: activation function
        act_config: kwarg dict for activation function
    """    
    def __init__(
            self,
            in_ch,
            out_ch=None,
            kernel_size=3,
            stride=1,
            batch_norm=True,
            bias=None,
            act='ReLU',
            act_config={}):
        super(SeparableConv2d, self).__init__()
        pass


class SeparableStack(object)
    """ stack of SeparableConv2d
    Args:
        in_ch<int>: number of input channels
        out_chs<list<int>|None>: list of output channels for each SeparableConv2d
        out_ch<int>/depth<int>:
            - use if out_chs is None
            - if out_ch is None out_ch=in_ch
            - create 'depth' number of layers (ie out_chs=[out_ch]*depth)
        res<bool>:
            - if true add resnet 'ident' to output of SeparableConv2d-Blocks.
            - if in_ch != out_ch perform 1x1 Conv to match channels
    """
    def __init__(self,
            in_ch,
            out_chs=None,
            out_ch=None,
            depth=1,
            res=False):
        super(SeparableStack, self).__init__()
        if not out_chs:
            if not out_ch:
                out_ch=in_ch
            out_chs=[out_ch]*depth
        self.sconvs=self._sconv_blocks(in_ch,out_chs)
        self.res=res
        self.ident_conv=self._ident_conv()


    def forward(self,x):
        xout=self.sconvs(x)
        if self.res:
            if self.ident_conv:
                x=self.ident_conv(x)
            return x.add_(xout)
        else:
            return xout


    #
    # INTERNAL
    #
    def _sconv_blocks(self,ch,depth):
        blocks=[]
        for ch in range(out_chs):
            blocks.append(SeparableConv2d(in_ch,ch))
            in_ch=ch
        return nn.ModuleList(blocks)


    def _ident_conv(self):
        if self.res and in_ch!=out_chs[-1]:
            return Conv(
                in_ch,
                out_ch
                kernel_size=1)
        else:
            return False




#
# XCEPTION ARCHITECTURE
#
class EntryBlock(nn.Module):
    """ Xception Entry Block:

    The first two layers of Xception network, before
    any SeparableConv2d blocks

    Args:
        in_ch<int>: number of input channels
        entry_ch<int>: out_ch of first block (the stride-2 block)
        entry_out_ch<int>: out_ch of second block
    """
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
    """ Xception Block:

    ResStack of SeparableConv2d blocks where the last
    SeparableConv2d and the "res_ident_block" have stride, 
    half-ing the dimensionality of the input

    Args:
        in_ch<int>: number of input channels
        out_ch<int>: 
            - number of output channels
            - if out_ch is None, out_ch = in_ch
        depth<int>: 
            - must be >= 2
            - total number of layers
            - the 3x3 stack then looks like:
                * SeparableConv2d(in_ch,out_ch)
                * SeparableConv2d(out_ch,out_ch) x (depth-2)
                * SeparableConv2d(in_ch,out_ch,stride=2)
    """
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
        if self.sconv_blocks_depth:
            self.sconv_blocks=SeparableStack(
                    out_ch,
                    depth=self.sconv_blocks_depth )
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
        return xpc.add_(x)





