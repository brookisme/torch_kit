import torch.nn as nn
import torch.nn.functional as F
import pytorch_nns.helpers as h
import pytorch_nns.models.xception.blocks as blocks



class Xception(nn.Module):
    r""" (Modified Aligned) Xception

    Xception Network or Backbone from DeeplabV3+
    
    Args:
        in_ch
        output_stride<int>: output_stride
        low_level_stride<int>:
            - the stride at which to return low-level-features
            - if None: low_level_stride = output_stride // 2
            - if False `forward` does not return low-level-features
        entry_ch: 32
        entry_out_ch: 64
        xblock_chs: [128,256,728]
        bottleneck_depth: 16
        exit_xblock_ch: 1024
        exit_stack_chs: [1536,1536,2048]
        xblock_depth: 3
        nb_classes<int|None>:
            - if None do not add... 
            - otherwise ...
    Properties:
    Links:

    """
    def __init__(self,
            in_ch,
            output_stride=16,
            low_level_stride=4,
            entry_ch=32,
            entry_out_ch=64,
            xblock_chs=[128,256,728],
            bottleneck_depth=16,
            exit_xblock_ch=1024,
            exit_stack_chs=[1536,1536,2048],
            xblock_depth=3,
            nb_classes=None):
        super(Xception,self).__init__()
        if low_level_stride is None:
            low_level_stride=int(output_stride//2)
        self.output_stride=output_stride
        self.low_level_stride=low_level_stride
        self._init_stride_state()
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
        if nb_classes:
            self.output_block=self._output_block(nb_classes)
        else:
            self.output_block=False


    def forward(self,x):
        self._init_stride_state()
        x=self.entry_block(x)
        for block in self.xblocks:
            x=self.xblocks(x)
            self._increment_stride_state()
            if self.stride_state==self.low_level_stride: 
                xlow=x
        x=self.bottleneck(x)
        x=self.exit_xblock(x)
        x=self.exit_stack(x)
        if self.output_block:
            x=self.output_block
        if self.low_level_stride:
            return x, xlow
        else:
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
        return nn.ModuleList(layers)


    def _init_stride_state(self):
        self.dilation=1
        self.stride_index=0
        self.stride_state=None


    def _increment_stride_state(self):
        self.stride_index+=1
        self.stride_state=(2**self.stride_index)
        if self.stride_state>=self.output_stride:
            self.dilation*=2


    def _output_block(self,nb_classes):
        """ TODO: Implement output block for Image Classification """
        pass
