import torch
import torch.nn as nn
import torch.nn.functional as F



class SqueezeExcitation(nn.Module):
    def __init__(self, nb_channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.nb_channels=nb_channels
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
                nn.Linear(nb_channels, nb_channels // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(nb_channels // reduction, nb_channels),
                nn.Sigmoid())

        
    def forward(self, x):
        y = self.avg_pool(x).view(-1,self.nb_channels)
        y = self.fc(y).view(-1,self.nb_channels,1,1)
        return x * y
    




class ConvBlock(nn.Module):

    def __init__(self,
            in_ch,
            in_size,
            depth=2, 
            kernel_size=3, 
            stride=1, 
            padding=0, 
            out_ch=None,
            bn=True,
            se=True,
            act='relu',
            act_kwargs={}):
        super(ConvBlock, self).__init__()
        self.out_ch=out_ch or 2*in_ch
        self._set_post_processes(self.out_ch,bn,se,act,act_kwargs)
        self._set_conv_layers(
            depth,
            in_ch,
            kernel_size,
            stride,
            padding)
        self.out_size=in_size-depth*2*((kernel_size-1)/2-padding)

        
    def forward(self, x):
        x=self.conv_layers(x)
        if self.bn:
            x=self.bn(x)
        if self.act:
            x=self._activation(x)
        if self.se:
            x=self.se(x)
        return x

    
    def _set_post_processes(self,out_channels,bn,se,act,act_kwargs):
        if bn:
            self.bn=nn.BatchNorm2d(out_channels)
        else:
            self.bn=False
        if se:
            self.se=SqueezeExcitation(out_channels)
        else:
            self.se=False
        self.act=act
        self.act_kwargs=act_kwargs

        
    def _set_conv_layers(
            self,
            depth,
            in_ch,
            kernel_size,
            stride,
            padding):
        layers=[]
        for index in range(depth):
            if index!=0:
                in_ch=self.out_ch
            layers.append(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=self.out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding))
        self.conv_layers=nn.Sequential(*layers)

        
    def _activation(self,x):
        return getattr(F,self.act,**self.act_kwargs)(x)

    


class DownBlock(nn.Module):
    
    def __init__(self,
            in_ch,
            in_size,
            out_ch=None,
            depth=2,
            padding=0,
            bn=True,
            se=True,
            act='relu',
            act_kwargs={}):
        super(DownBlock, self).__init__()
        self.out_size=(in_size//2)-depth*(1-padding)*2
        self.out_ch=out_ch or in_ch*2
        self.down=nn.MaxPool2d(kernel_size=2)
        self.conv_block=ConvBlock(
            in_ch=in_ch,
            out_ch=self.out_ch,
            in_size=in_size//2,
            depth=depth,
            padding=padding,
            bn=bn,
            se=se,
            act=act,
            act_kwargs=act_kwargs)

        
    def forward(self, x):
        x=self.down(x)
        return self.conv_block(x)



class UpBlock(nn.Module):
    
    @staticmethod
    def cropping(skip_size,size):
        return (skip_size-size)//2
    
    
    def __init__(self,
            in_ch,
            in_size,
            out_ch=None,
            bilinear=False,
            crop=None,
            depth=2,
            padding=0,
            bn=True,
            se=True,
            act='relu',
            act_kwargs={}):
        super(UpBlock, self).__init__()
        self.crop=crop
        self.padding=padding
        self.out_size=(in_size*2)-depth*(1-padding)*2
        self.out_ch=out_ch or in_ch//2
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv_block=ConvBlock(
            in_ch,
            self.out_size,
            out_ch=self.out_ch,
            depth=depth,
            padding=padding,
            bn=bn,
            se=se,
            act=act,
            act_kwargs=act_kwargs)
        
        
    def forward(self, x, skip):
        x = self.up(x)
        skip = self._crop(skip,x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv_block(x)
        return x

    
    def _crop(self,skip,x):
        if self.padding is 0:
            if self.crop is None:
                self.crop=self.cropping(skip.size()[-1],x.size()[-1])
            skip=skip[:,:,self.crop:-self.crop,self.crop:-self.crop]
        return skip
