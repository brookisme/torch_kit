import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):


    def __init__(self,
            in_ch,
            out_ch=None,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
            pointwise_in=True,
            dropout=False):
        super(SeparableConv2d, self).__init__()
        out_ch=out_ch or in_ch
        self.pointwise_in=pointwise_in
        if self.pointwise_in:
            conv_ch=out_ch
        else:
            conv_ch=in_ch
        self.conv=nn.Conv2d(
            conv_ch, 
            conv_ch, 
            kernel_size, 
            stride, 
            padding, 
            dilation, 
            groups=conv_ch, 
            bias=bias)
        self.pointwise=nn.Conv2d(
            in_ch, 
            out_ch, 
            kernel_size=1, 
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias)
        if dropout:
            self.dropout=nn.Dropout2d(p=dropout)
        else:
            self.dropout=False


    def forward(self, x):
        if self.pointwise_in:
            x=self.pointwise(x)
            x=self.conv(x)
        else:
            x=self.conv(x)
            x=self.pointwise(x)
        if self.dropout:
            x=self.dropout(x)
        return x




class SeparableBlock(nn.Module):


    def __init__(self,
            depth,
            in_ch,
            out_ch=None,
            batch_norm=True,
            act='ReLU',
            act_config={},
            act_in=True,
            **config):
        super(SeparableBlock, self).__init__()
        self.in_ch=in_ch
        self.out_ch=out_ch or in_ch
        self.batch_norm=batch_norm
        self.layers=self._get_layers(depth,act,act_config,act_in,config)


    def forward(self, x):
        return self.layers(x)


    def _get_layers(self,depth,act,act_config,act_in,config):
        layers=[]
        for i in range(depth):
            if i or act_in:
                layers.append(self._act_layer(act)(**act_config))
            layers.append(
                SeparableConv2d(
                    self._in_ch(i),
                    self.out_ch,
                    **config))
            if self.batch_norm:
                layers.append(nn.BatchNorm2d(self.out_ch))
        return nn.Sequential(*layers)


    def _act_layer(self,act):
        if isinstance(act,str):
            return getattr(nn,act)
        else:
            return act


    def _in_ch(self,index):
        if index:
            in_ch=self.out_ch
        else:
            in_ch=self.in_ch
        return in_ch




class XDown(nn.Module):


    def __init__(self,
            depth,
            in_ch,
            out_ch=None,
            batch_norm=True,
            res_batch_norm=True,
            act='ReLU',
            act_config={},
            act_in=True,
            pool_kernel=3,
            pool_stride=2,
            **config):
        super(XDown, self).__init__()
        self.separable_block=SeparableBlock(
            depth=depth,
            in_ch=in_ch,
            out_ch=out_ch,
            batch_norm=batch_norm,
            act=act,
            act_config=act_config,
            act_in=act_in,
            **config)
        self.max_pooling=nn.MaxPool2d(
            pool_kernel, 
            pool_stride,padding=(pool_kernel-pool_stride))
        self.res_conv=nn.Conv2d(
            in_ch, 
            out_ch, 
            kernel_size=1, 
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            bias=False)
        if res_batch_norm:
            self.res_bn=nn.BatchNorm2d(out_ch)
        else:
            self.res_bn=False


    def forward(self, x):
        skip=self.res_conv(x)
        if self.res_bn:
            skip=self.res_bn(skip)
        x=self.separable_block(x)
        x=self.max_pooling(x)
        return skip.add_(x)



class XUp(nn.Module):


    def __init__(self,
            depth,
            in_ch,
            out_ch=None,
            batch_norm=True,
            act='ReLU',
            act_config={},
            act_in=True,
            pool_kernel=3,
            pool_stride=2,
            bilinear=False,
            **config):
        super(XUp, self).__init__()
        out_ch=out_ch or in_ch
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, 
                mode='bilinear', 
                align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                int(in_ch),
                int(out_ch),
                2,
                stride=2)
        self.separable_block=SeparableBlock(
            depth = depth,
            in_ch = out_ch*2,
            out_ch = out_ch,
            batch_norm = batch_norm,
            act = act,
            act_config = act_config,
            act_in = act_in,
            **config)


    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.separable_block(x)


