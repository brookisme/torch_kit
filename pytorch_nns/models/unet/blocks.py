import torch
import torch.nn as nn
from pytorch_nns.models.shared.blocks import Conv 
from abc import ABCMeta, abstractmethod

PADDING=0

class UNetBlock(nn.Module,metaclass=ABCMeta):
    """ this is an abstract class to ensure all UNetBlocks
        Up,Down,Bottleneck have the signature:
        in_ch,out_ch,**config
    """
    @abstractmethod
    def __init__(self,in_ch,out_ch,**config):
        super(UNetBlock, self).__init__()
        pass




class Down(UNetBlock):
    r""" UNet Down Block

    Layers:
        1. MaxPooling
        2. Conv Block 

    Args:
        config: Conv kwargs
    """    
    def __init__(self,in_ch,out_ch=None,**config):
        super(Down, self).__init__(in_ch,out_ch)
        self.down=nn.MaxPool2d(kernel_size=2)
        self.conv_block=Conv(in_ch=in_ch,out_ch=out_ch,**config)


    def forward(self, x):
        x=self.down(x)
        return self.conv_block(x)




class Up(UNetBlock):
    r""" UNet Up Block

    Layers:
        1. ConvTranspose2d or bilinear Upsample
        2. Concat with (possibly cropped) Skip connection
        3. Conv Block 

    Args:
        in_ch (int): Number of channels in input
        out_ch (int <None>): Number of channels in output (if None => in_ch)
        bilinear (bool <False>): If true use bilinear Upsample otherwise ConvTranspose2d
        crop (int <int or None>): 
            If padding is 0: cropping for skip connection.  If None the cropping will be
            calculated.  Note: both input-size minus skip-size must be even. 
        config: Conv kwargs
    """      
    @staticmethod
    def cropping(skip_size,size):
        r""" calculates crop size

        Args:
            skip_size (int): size (=h=w) of skip connection
            size (int): size (=h=w) of input
        """
        return int((skip_size-size)//2)
    
 
    def __init__(self,
            in_ch,
            out_ch=None,
            out_up_ch=None,
            bilinear=False,
            crop=None,
            **config):
        super(Up, self).__init__(in_ch,out_ch)
        self.crop=crop
        out_ch=out_ch or in_ch//2
        out_up_ch=out_up_ch or out_ch
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(int(in_ch),int(out_up_ch),2,stride=2)
        self.conv_block=Conv(
            in_ch=in_ch,
            out_ch=out_ch,
            **config)

        
    def forward(self, x, skip):
        x = self.up(x)
        skip = self._crop(skip,x)
        x = torch.cat([skip, x], dim=1)
        return self.conv_block(x)

    
    def _crop(self,skip,x):
        if self.conv_block.padding is 0:
            if self.crop is None:
                self.crop=self.cropping(skip.size()[-1],x.size()[-1])
            skip=skip[:,:,self.crop:-self.crop,self.crop:-self.crop]
        return skip