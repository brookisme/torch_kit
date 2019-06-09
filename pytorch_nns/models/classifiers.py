import sys
import re
import torch.nn as nn
from pytorch_nns.models.helpers import output_activation,to_camel
import pytorch_nns.models.blocks as blocks


#
# CONSTANTS
#
self=sys.modules[__name__]
ALIAS={
    'gap': 'GAPClassifier'
}
NB_CLASSES_ERROR='GAPClassifier: dense_chs[-1] must equal nb_classes'



#
# HELPERS 
#
def get(classifier):
    if isinstance(classifier,str):
        class_name=ALIAS.get(classifier)
        if not class_name:
            class_name=to_camel(classifier)
            class_name=re.sub('Gap','GAP',class_name)
        classifier=getattr(self,class_name)
    return classifier




#
# CLASSIFIERS
#
class GAPClassifier(nn.Module):
    """ GlobalAveragePooling Classification Block

    1. ConvBlocks
    2. GAP
    3. DenseBlocks
    4. Activation
    
    Args:
        in_ch,
        nb_classes,
        nb_convs=1,
        conv_out_ch=None,
        conv_chs=None,
        conv_config={},
        pooling=AVERAGE,
        nb_dense=1,
        dense_chs=None,
        dense_config={},
        act=None,
        act_config={}
    """
    #
    # CONSTANTS
    #
    AVERAGE='avg'
    MAX='max'


    #
    # PUBLIC
    #
    def __init__(
            self,
            in_ch,
            nb_classes,
            nb_convs=1,
            conv_out_ch=None,
            conv_chs=None,
            conv_config={},
            pooling=AVERAGE,
            nb_dense=1,
            dense_chs=None,
            dense_config={},
            act=None,
            act_config={}):
        super(GAPClassifier, self).__init__()
        if nb_convs:
            conv_out_ch=conv_out_ch or nb_classes
            self.convs=blocks.Conv(
                in_ch=in_ch,
                out_ch=conv_out_ch,
                out_chs=conv_chs,
                depth=nb_convs,
                **conv_config)
        else:
            self.convs=False
        self.pooling=self._pooling(pooling)
        if nb_dense:
            if conv_chs:
                dense_in_ch=conv_chs[-1]
            else:
                dense_in_ch=conv_out_ch
            if dense_chs:
                if dense_chs[-1]!=nb_classes:
                    raise ValueError(NB_CLASSES_ERROR)
            self.dense=blocks.Dense(
                in_ch=dense_in_ch,
                out_ch=nb_classes,
                out_chs=dense_chs,
                depth=nb_dense,
                **dense_config)
        else:
            self.dense=False
        self.act=output_activation(act,nb_classes,**act_config)
        

    def forward(self,x):
        if self.convs:
            x=self.convs(x)
        x=self.pooling(x)
        x=x.view(-1,x.shape[1])
        if self.dense:
            x=self.dense(x)
        if self.act:
            x=self.act(x)
        return x



    #
    # INTERNAL
    #
    def _pooling(self,pooling):
        if pooling:
            if pooling==GAPClassifier.AVERAGE:
                pooling=nn.AdaptiveAvgPool2d((1,1))
            elif pooling==GAPClassifier.MAX:
                pooling=nn.AdaptiveMaxPool2d((1,1))
        return pooling


