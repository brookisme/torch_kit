import torch
import numpy as np
import pickle
import json
from datetime import datetime


#
# CONFIG
#
ACT_ERROR_TMPL='[ERROR] unet: {} not implemented via str'
TIMESTAMP_FMT="%H:%M:%S (%Y-%m-%d)"
CUDA='cuda'
CPU='cpu'


#
# Python
#
def read_json(path):
    with open(path,'rb') as f:
        data=json.load(f)
    return data


def get_time():
    now=datetime.now()
    now_str=now.strftime(TIMESTAMP_FMT)
    return now,now_str


def save_pickle(obj,path):
    """ save object to pickle file
    """    
    with open(path,'wb') as file:
        pickle.dump(obj,file,protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path):
    """ read pickle file
    """    
    with open(path,'rb') as file:
        obj=pickle.load(file)
    return obj


def get_index(value,value_list):
    if isinstance(value,int):
        return value
    else:
        return value_list.index(value)


#
# OUTPUT
#
def print_line(char='-',length=75):
    print(char*length)



#
# PyTorch
#
def to_numpy(tensor):
    if torch.is_tensor(tensor):
        tensor=tensor.cpu().detach().numpy()
    return tensor


def to_tensor(v):
    if isinstance(v,list):
        v=torch.tensor(v)
    elif isinstance(v,np.ndarray):
        v=torch.from_numpy(v)
    return v


def argmax(tsr_arr,axis=0):
    if torch.is_tensor(tsr_arr):
        tsr_arr=torch.argmax(tsr_arr,dim=axis)
    else:
        tsr_arr=np.argmax(tsr_arr,axis=axis)
    return tsr_arr


def get_device(force_cpu=False):
    use_gpu=((not force_cpu) and torch.cuda.is_available())
    return torch.device(CUDA if use_gpu else CPU)


def load_weights(model,weights_path,device=None,map_location=None):
    if not map_location:
        if device and (device.type=='cpu'):
            map_location='cpu'
    state_dict=torch.load(weights_path,map_location=map_location)
    model.load_state_dict(state_dict)


def get_model(
        net,
        config={},
        weight_initializer=None,
        init_weights=None,
        device=None,
        map_location=None,
        seed=None):
    if seed:
        torch.manual_seed(seed)
    if device=='get':
        device=get_device()
    model=net(**config)
    if init_weights:
        print("init_weights: ",init_weights)
        load_weights(model,init_weights,device=device,map_location=map_location)
    if device:
        model=model.to(device)
    if weight_initializer:
        print("weight_initializer: ",weight_initializer)
        model.apply(weight_initializer)
    return model


def get_output_activation(output_activation=None,out_ch=None,**act_kwargs):
    if isinstance(output_activation,str):
        if output_activation.lower()=='sigmoid':
            act=nn.Sigmoid()
        elif output_activation.lower()=='softmax':
            act=nn.Softmax(dim=1)
        elif output_activation.lower()=='log_softmax':
            act=nn.LogSoftmax(dim=1)
        else:
            raise ValueError(ACT_ERROR_TMPL.format(output_activation))
    elif output_activation is None:
        if out_ch==1:
            act=nn.Sigmoid()
        else:
            act=nn.Softmax(dim=1)
    elif output_activation is False:
        act=False
    else:
        if callable(output_activation()):
            act=output_activation(**act_kwargs)
        else:
            act
    return act 


def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad=False
        

def freeze_layers(model,end=None,start=None):
    layer_list=list(model.children())
    for layer in layer_list[start:end]:
        freeze_layer(layer)


def trainable_params(model):
    return filter(lambda p: p.requires_grad, model.parameters())


def nb_trainable_params(model):
    return sum([np.prod(p.size()) for p in trainable_params(model)])
