import torch.nn as nn
import torch.nn.functional as F


RES_MULTIPLIER=0.75
PADDING=0


class Conv(nn.Module):
    r""" Conv Block
    
    Args:
        in_ch (int): Number of channels in input
        out_ch (int <None>): Number of channels in output (if None => in_ch)
        depth (int <2>): The number of convolutional layers 
        kernel_size (int <3>): Kernel Size
        stride (int <1>): Stride
        padding (int|str <0>): int or same if padding='same' -> int((kernel_size-1)/2) 
        batch_norm (bool <True>): Add batch norm after conv
        dropout (False|float <False>): Dropout to be applied after Conv
        act (str <'relu'>): Method name of activation function after each Conv Layer
        act_config (dict <{}>): Kwargs for activation function after each Conv Layer
    """
    #
    # CONSTANTS
    #
    SAME='same'


    #
    # STATIC METHODS
    #
    @staticmethod
    def same_padding(kernel_size):
        r""" calculates same padding size
        """
        return (kernel_size-1)//2


    #
    # PUBLIC METHODS
    #
    def __init__(self,
            in_ch,
            out_ch=None,
            depth=2, 
            kernel_size=3, 
            stride=1, 
            padding=PADDING, 
            batch_norm=True,
            dropout=False,
            act='ReLU',
            act_config={}):
        super(Conv, self).__init__()
        self.in_ch=int(in_ch)
        self.out_ch=int(out_ch) or self.in_ch
        if padding==Conv.SAME:
            padding=Conv.same_padding(kernel_size)
        self.padding=padding
        self.conv_layers=self._conv_layers(
            depth,
            kernel_size,
            stride,
            padding,
            batch_norm,
            dropout,
            act,
            act_config)

        
    def forward(self, x):
        return self.conv_layers(x)


    #
    # INTERNAL METHODS
    #    
    def _conv_layers(self,
            depth,
            kernel_size,
            stride,
            padding,
            batch_norm,
            dropout,
            act,
            act_config):
        layers=[]
        for index in range(depth):
            if index==0:
                in_ch=self.in_ch
            else:
                in_ch=self.out_ch
            layers.append(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=self.out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=(not batch_norm)))
            if batch_norm:
                layers.append(nn.BatchNorm2d(self.out_ch))
            if act:
                layers.append(self._act_layer(act)(**act_config))
            if dropout:
                layers.append(nn.Dropout2d(p=dropout))
        return nn.Sequential(*layers)

        
    def _act_layer(self,act):
        if isinstance(act,str):
            return getattr(nn,act)
        else:
            return act


RES_MULTIPLIER=0.75
class Residule(nn.Module):
    r""" Residule

    Args:
        block: block or list of layers
        multiplier <float [RES_MULTIPLIER]>: ident multiplier
        crop: <int|bool> 
            if <int> cropping = crop
            elif True calculate cropping
            else no cropping
    Links:
        TODO: GOT IDEA FROM FASTAI SOMEWHERE

    """
    def __init__(self,
            block,
            in_ch,
            out_ch=None,
            multiplier=RES_MULTIPLIER,
            crop=False):
        super(Residule, self).__init__()
        self.block=self._process_block(block)
        self.in_ch=in_ch
        self.out_ch=out_ch or in_ch
        self.multiplier=multiplier
        self.crop=crop
        if self.in_ch!=self.out_ch:        
            self.ident_conv=nn.Conv2d(
                        in_channels=self.in_ch,
                        out_channels=self.out_ch,
                        kernel_size=1)
        else:
            self.ident_conv=False

    
    def forward(self, x):
        block_out=self.block(x)
        if self.crop:
            x=self._crop(x,block_out)
        if self.ident_conv:
            x=self.ident_conv(x)
        return (self.multiplier*x).add_(block_out)


    def _process_block(self,block):
        if isinstance(block,list):
            return nn.Sequential(*block)
        else:
            return block


    def _crop(self,x,layers_out):
        if isinstance(self.crop,bool):
            # get cropping
            out_size=layers_out.size()[-1]
            x_size=x.size()[-1]
            self.crop=(x_size-out_size)//2
        return x[:,:,self.crop:-self.crop,self.crop:-self.crop]



class SqueezeExcitation(nn.Module):
    r""" Squeeze and Excitation Block

    Args:
        nb_ch (int): Number of Channels in input image
        
    Links:
        https://arxiv.org/abs/1709.01507

    """
    def __init__(self, nb_ch, reduction=16, warn=True):
        super(SqueezeExcitation, self).__init__()
        self.nb_ch=nb_ch
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.reduction_ch=nb_ch//reduction
        if self.reduction_ch:
            self.fc=nn.Sequential(
                    nn.Linear(nb_ch,self.reduction_ch),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.reduction_ch, nb_ch),
                    nn.Sigmoid())
        elif warn:
            print('[WARNING] SqueezeExcitation skipped. nb_ch < reduction')

        
    def forward(self, x):
        if self.reduction_ch:
            y = self.avg_pool(x).view(-1,self.nb_ch)
            y = self.fc(y).view(-1,self.nb_ch,1,1)
            return x * y
        else:
            return x



class GeneralizedResBlock(nn.Module):
    r""" GeneralizedResBlock

    Args:
        blocks: list of blocks or lists of layers
        multiplier <float [RES_MULTIPLIER]>: ident multiplier
        crop: <int|bool> 
            if <int> cropping = crop
            elif True calculate cropping
            else no cropping

    """
    def __init__(self,
            blocks,
            in_ch,
            out_ch,
            multiplier=RES_MULTIPLIER,
            crop=True):
        super(GeneralizedResBlock, self).__init__()
        self.blocks=self._process_blocks(blocks)
        self.in_ch=in_ch
        self.out_ch=out_ch
        self.multiplier=multiplier
        self.crop=crop
        if self.in_ch!=self.out_ch:        
            self.ident_conv=nn.Conv2d(
                        in_channels=self.in_ch,
                        out_channels=self.out_ch,
                        kernel_size=1)
        else:
            self.ident_conv=False

    
    def forward(self, x):
        x_out=self.blocks[0](x)
        for b in self.blocks[1:]:
            x_out+=b(x)
        if self.crop:
            x=self._crop(x,x_out)
        if self.ident_conv:
            x=self.ident_conv(x)
        return (self.multiplier*x) + x_out 


    def _process_blocks(self,blocks):
        return nn.ModuleList(self._process_block(b) for b in blocks)


    def _process_block(self,block):
        if isinstance(block,list):
            return nn.Sequential(*block)
        else:
            return block


    def _crop(self,x,layers_out):
        if self.crop is True:
            # get cropping
            out_size=layers_out.size()[-1]
            x_size=x.size()[-1]
            self.crop=(x_size-out_size)//2
        return x[:,:,self.crop:-self.crop,self.crop:-self.crop]



class GeneralizedConvResnet(nn.Module):
    r""" GenConvs: for GeneralizedResBlocks
    """
    def __init__(self,
                in_ch,
                out_ch,
                in_size,
                kernel_sizes=[3,5], 
                paddings=None,
                multiplier=RES_MULTIPLIER,
                crop=True,
                **conv_kwargs):
        super(GeneralizedConvResnet, self).__init__()
        conv_kwargs.pop('padding',None)    
        self.in_ch=in_ch
        self.out_ch=out_ch
        if (paddings is None) or (paddings is 'same'):
            paddings=['same']*len(kernel_sizes)
            self.cropping=False
        elif isinstance(paddings,int):
            k0=kernel_sizes[0]
            p0=paddings
            same_padding=Conv.same_padding(k0)
            self.cropping=conv_kwargs['depth']*(same_padding-p0)
            paddings=[p0]
            for k in kernel_sizes[1:]:
                p=((k-k0)//2)-p0
                paddings.append(p)
        self.kernel_sizes=kernel_sizes
        self.paddings=paddings
        convs=[]
        for k,p in zip(kernel_sizes,paddings):
            conv=Conv(
                in_ch=in_ch,
                out_ch=out_ch,
                in_size=in_size,
                kernel_size=k,
                padding=p,
                **conv_kwargs)
            convs.append(conv)
        self.out_size=convs[-1].out_size
        self.gen_res_block=GeneralizedResBlock(
                convs,
                in_ch,
                out_ch,
                multiplier=multiplier,
                crop=crop )


    def forward(self, x):
        return self.gen_res_block(x)

