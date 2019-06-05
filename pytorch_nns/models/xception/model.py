import torch.nn as nn
import torch.nn.functional as F
import pytorch_nns.helpers as h
import pytorch_nns.models.xception.blocks as blocks



class Xception(nn.Module):
    r""" (Modified Aligned) Xception

    Xception Backbone from DeeplabV3+
    
    Args:
        in_ch
        entry_ch: 32
        entry_out_ch: 64
        xblock_chs: [128,256,728]
        bottleneck_depth: 16
        exit_xblock_ch: 1024
        exit_stack_chs: [1536,1536,2048]
        xblock_depth: 3
        
    Properties:
    Links:

    """
    def __init__(self,
            in_ch,
            entry_ch=32,
            entry_out_ch=64,
            xblock_chs=[128,256,728],
            bottleneck_depth=16,
            exit_xblock_ch=1024,
            exit_stack_chs=[1536,1536,2048],
            xblock_depth=3):
        super(Xception,self).__init__()
        self.entry_block=blocks.EntryBlock(in_ch,entry_ch,entry_out_ch)
        self.xblocks=self._xblocks(entry_out_ch,xblock_chs,xblock_depth)
        self.bottleneck=blocks.SeparableResStack(
            ch=xblock_chs[-1],
            depth=bottleneck_depth )
        self.exit_xblock=blocks.XBlock(
                in_ch=xblock_chs[-1],
                out_ch=exit_xblock_ch,
                dept=xblock_depth )
        self.exit_stack=blocks.SeparableStack(*exit_stack_chs)


    def forward(self,x):
        x=self.entry_block(x)
        x=self.xblocks(x)
        x=self.bottleneck(x)
        x=self.exit_xblock(x)
        x=self.exit_stack(x)
        return x


    #
    # INTERNAL
    #
    def _xblocks(self,in_ch,xblock_chs,xblock_depth):
        blocks=[]
        for ch in xblock_chs:
            blocks.append(blocks.XBlock(in_ch,ch,xblock_depth))
            in_ch=ch
        return nn.ModuleList(blocks)






