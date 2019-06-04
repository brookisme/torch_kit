import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self,
            in_ch,
            out_ch=None,
            kernel_size=3,
            stride=1,
            bias=False,
            batch_norm=True,
            act='relu',
            act_kwargs={}):
        super(SeparableConv2d, self).__init__()
        pass


class  XBlock(object):
    """ entry: 128 256 728 """
    def __init__(self,
            in_ch,
            out_ch=None,
            depth=3
        ):
        super(XBlock, self).__init__()
        pass
        

class SeparableResStack(object):
    """  bottleneck """
    def __init__(self,
            ch,
            depth=16,
        ):
        super(SeparableResStack, self).__init__()
        pass        



class SeparableStack(object)
    """ exit """
    def __init__(self,in_ch,*out_chs):
        super(ConvStack, self).__init__()
        pass  