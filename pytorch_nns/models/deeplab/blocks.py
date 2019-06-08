import torch
import torch.nn as nn
from pytorch_nns.models.helpers import activation,same_padding


#
# CONSTANTS
#
DEFAULT_DROPOUT=0.5
CROP_TODO="TODO: Need to crop 1x1 Conv to implement non-same padding"


#
# GENERAL BLOCKS
#
class ASPP(nn.Module):
    """ Atrous Spatial Pyramid Pooling

    Args:
        in_ch<int>: number of input channels
        out_ch<int|None>: if out_ch is None out_ch=in_ch
        kernel_sizes<list[int]>: kernel_size for each conv layer
        dilations<list[int]>: dilation for each conv layer
        pooling<bool>: include image_pooling block

    """ 
    #
    # CONSTANTS
    #
    SAME='same'
    RELU='ReLU'
    AVERAGE='avg'
    MAX='max'



    #
    # INSTANCE METHODS
    #
    def __init__(
            self,
            in_ch,
            out_ch=256,
            kernel_sizes=[1,3,3,3],
            dilations=[1,6,12,18],
            pooling=AVERAGE,
            batch_norm=True,
            bias=None,
            out_kernel_size=1,
            out_batch_norm=True,
            act=RELU,
            act_config={},
            dropout=False):
        super(ASPP, self).__init__()
        if out_ch is None:
            out_ch=in_ch
        self.in_ch=in_ch
        self.out_ch=out_ch
        self.nb_aconvs=len(kernel_sizes)
        if not bias:
            bias=(not batch_norm)
        self.batch_norm=batch_norm
        self.bias=bias
        self.pooling=self._pooling(pooling)
        self.batch_norms=self._batch_norms(batch_norm)
        self.aconv_list=self._aconv_list(kernel_sizes,dilations)
        self.out_conv=self._out_conv(out_kernel_size,out_batch_norm)
        self.act=activation(act,**act_config)


    def forward(self, x):
        stack=[l(x) for l in self.aconv_list]
        if self.pooling:
            stack.append(self.pooling(x)*torch.ones(stack[0].shape))
        x=torch.cat(stack,dim=1)
        x=self.out_conv(x)
        if self.act:
            x=self.act(x)
        return x


    def _aconv_list(self,kernels,dilations):
        aconvs=[self._aconv(k,d,i) for i,(k,d) in enumerate(zip(kernels,dilations))]
        return nn.ModuleList(aconvs)


    def _aconv(self,kernel,dilation,index):
        aconv=nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=self.out_ch,
            kernel_size=kernel,
            dilation=dilation,
            padding=same_padding(kernel,dilation),
            bias=self.bias)
        layers=[aconv]
        if self.batch_norms:
            layers.append(self.batch_norms[index])
        # RELU?
        return nn.Sequential(*layers)


    def _pooling(self,pooling):
        if pooling:
            if pooling==ASPP.AVERAGE:
                pooling=nn.AdaptiveAvgPool2d((1,1))
            elif pooling==ASPP.MAX:
                pooling=nn.AdaptiveMaxPool2d((1,1))
            else:
                raise NotImplementedError("pooling must be 'avg' or 'max'")
            pooling=nn.Sequential(
                pooling,
                nn.Conv2d(
                    self.in_ch, 
                    self.out_ch, 
                    kernel_size=1, 
                    stride=1,
                    bias=False),
                nn.BatchNorm2d(self.out_ch),
                nn.ReLU())
            # REMOVE RELU?
        else:
            pooling=False
        return pooling


    def _batch_norms(self,batch_norm):
        if batch_norm:
            batch_norms=[ nn.BatchNorm2d(self.out_ch) for _ in range(self.nb_aconvs) ]
        else:
            batch_norms=False
        return batch_norms


    def _out_conv(self,kernel_size,batch_norm):
        in_ch=self.nb_aconvs*self.out_ch
        if self.pooling: in_ch+=self.out_ch
        conv=nn.Conv2d(
            in_channels=in_ch,
            out_channels=self.out_ch,
            kernel_size=kernel_size,
            padding=same_padding(kernel_size))
        layers=[conv]
        if batch_norm:
            layers.append(nn.BatchNorm2d(self.out_ch))
        return nn.Sequential(*layers)



