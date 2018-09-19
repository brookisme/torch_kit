import torch
import numpy as np
import pickle
import json
from datetime import datetime


#
# CONFIG
#
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


def argmax(tsr_arr,dim=0):
    if torch.is_tensor(tsr_arr):
        tsr_arr=torch.argmax(tsr_arr,dim=dim)
    else:
        tsr_arr=np.argmax(tsr_arr,axis=dim)
    return tsr_arr


def get_device(force_cpu=False):
    use_gpu=((not force_cpu) and torch.cuda.is_available())
    return torch.device(CUDA if use_gpu else CPU)


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
        if not map_location:
            if device and (device.type=='cpu'):
                map_location='cpu'
        print("init_weights: ",init_weights)
        model.load_state_dict(torch.load(init_weights,map_location=map_location))
    if device:
        model=model.to(device)
    if weight_initializer:
        print("weight_initializer: ",weight_initializer)
        model.apply(weight_initializer)
    return model



