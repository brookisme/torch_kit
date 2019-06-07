import torch.nn as nn
import torch.nn.functional as F
import pytorch_nns.helpers as h
import pytorch_nns.models.xception.blocks as blocks



class Xception(nn.Module):
    r""" (Modified Aligned) Xception

    Xception Backbone from DeeplabV3+
    
    Args:
        in_ch
        output_stride<int>: output_stride
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
            output_stride=16,
            entry_ch=32,
            entry_out_ch=64,
            xblock_chs=[128,256,728],
            bottleneck_depth=16,
            exit_xblock_ch=1024,
            exit_stack_chs=[1536,1536,2048],
            xblock_depth=3):
        super(Xception,self).__init__()
        self._init_stride_state(output_stride)
        self.entry_block=blocks.EntryBlock(in_ch,entry_ch,entry_out_ch)
        self._increment_stride_state()
        self.xblocks=self._xblocks(
            entry_out_ch,
            xblock_chs,
            xblock_depth)
        self.bottleneck=blocks.SeparableStack(
            in_ch=xblock_chs[-1],
            depth=bottleneck_depth,
            res=True,
            dilation=self.dilation)
        self.exit_xblock=blocks.XBlock(
                in_ch=xblock_chs[-1],
                out_ch=exit_xblock_ch,
                depth=xblock_depth,
                dilation=self.dilation)
        self._increment_stride_state()
        self.exit_stack=blocks.SeparableStack(
            in_ch=exit_xblock_ch,
            out_chs=exit_stack_chs,
            dilation=self.dilation)


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
    def _xblocks(self,in_ch,out_ch_list,depth):
        layers=[]
        for ch in out_ch_list:
            layers.append(blocks.XBlock(
                in_ch,
                out_ch=ch,
                depth=depth,
                dilation=self.dilation,
            ))
            self._increment_stride_state()
            in_ch=ch
        return nn.Sequential(*layers)


    def _init_stride_state(self,output_stride):
        self.output_stride=output_stride
        self.stride_index=0
        self.dilation=1
    

    def _increment_stride_state(self):
        self.stride_index+=1
        if ((2**self.stride_index)>=self.output_stride):
            self.dilation*=2


