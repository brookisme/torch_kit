PADDING=0
import torch
import torch.nn as nn
from pytorch_nns.models.shared.blocks import Conv 


class Down(nn.Module):
    r""" UNet Down Block

    Layers:
        1. MaxPooling
        2. Conv Block 

    Args:
        Conv kwargs (See Conv)
    """    
    def __init__(self,
            **kwargs):
        super(Down, self).__init__()
        self.down=nn.MaxPool2d(kernel_size=2)
        self.conv_block=Conv(**kwargs)

    def forward(self, x):
        x=self.down(x)
        return self.conv_block(x)




class Up(nn.Module):
    r""" UNet Up Block

    Layers:
        1. ConvTranspose2d or bilinear Upsample
        2. Concat with (possibly cropped) Skip connection
        3. Conv Block 

    Args:
        in_ch (int): Number of channels in input
        out_ch (int <None>): Number of channels in output (if None => in_ch)
        depth (int <2>): The number of convolutional layers 
        bilinear (bool <False>): If true use bilinear Upsample otherwise ConvTranspose2d
        crop (int <int or None>): 
            If padding is 0: cropping for skip connection.  If None the cropping will be
            calculated.  Note: both input-size minus skip-size must be even. 
        padding (int <0>): Padding. Use 0 or 1 since the kernal size is fixed at 3x3
        batch_norm (bool <True>): Add batch norm after conv
        act (str <'relu'>): Method name of activation function after Conv Block
        act_kwargs (dict <{}>): Kwargs for activation function after Conv Block
        
    Properties:
        out_ch <int>: Number of channels of the output
        out_size <int>: Size (=H=W) of input
        cropping <int>: Cropping for skip connection
        padding <int>: Padding for conv block

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
            depth=2,
            bilinear=False,
            crop=None,
            padding=PADDING,
            kernel_size=3,
            batch_norm=False,
            act='ReLU',
            act_config={}):
        super(Up, self).__init__()
        self.crop=crop
        out_ch=out_ch or in_ch//2
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(int(in_ch),int(out_ch),2,stride=2)
        self.conv_block=Conv(
            in_ch=in_ch,
            out_ch=out_ch,
            depth=depth,
            padding=padding,
            kernel_size=kernel_size,         
            batch_norm=batch_norm,
            act=act,
            act_config=act_config)

        
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