import torch.nn as nn
import torch.nn.functional as F
import pytorch_nns.helpers as h
import pytorch_nns.models.xception.blocks as blocks



class Xception(nn.Module):
    r""" (Modified Aligned) Xception

    Xception Backbone from DeeplabV3+
    
    Args:
        entry_ch=32,
        entry_out_ch=64,
        xblock_chs=[128,256,728],
        bottleneck_depth=16,
        exit_xblock_ch=1024,
        exit_stack_chs=[1536,1536,2048]
    Properties:
    Links:

    """
    def __init__(self,
            entry_ch=32,
            entry_out_ch=64,
            xblock_chs=[128,256,728],
            bottleneck_depth=16,
            exit_xblock_ch=1024,
            exit_stack_chs=[1536,1536,2048]):
        super(Xception,self).__init__()
        self.entry_block
        self.xblocks
        self.bottleneck
        self.exit_xblock
        self.exit_stack


    def forward(self,x):
        x=self.entry_block
        x=self.xblocks
        x=self.bottleneck
        x=self.exit_xblock
        x=self.exit_stack
        return x