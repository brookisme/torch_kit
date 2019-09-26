import torch
import torch.nn as nn
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
    if map_location is None:
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


def freeze_layer(layer,unfreeze=False):
    for param in layer.parameters():
        param.requires_grad=unfreeze


def freeze_layers(model,unfreeze=False,end=None,start=None):
    layer_list=list(model.children())
    for layer in layer_list[start:end]:
        freeze_layer(layer,unfreeze=unfreeze)


def trainable_params(model):
    return filter(lambda p: p.requires_grad, model.parameters())


def nb_trainable_params(model):
    return sum([np.prod(p.size()) for p in trainable_params(model)])
