import torch
import numpy as np
import pickle

#
# CONFIG
#
CUDA='cuda'
CPU='cpu'


#
# Python
#
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


#
# OUTPUT
#
def print_line(char='-',length=75):
    print(char*length)


#
# PyTorch
#
def to_tensor(v):
    if isinstance(v,list):
        v=torch.tensor(v)
    elif isinstance(v,np.ndarray):
        v=torch.from_numpy(v)
    return v



def get_device(force_cpu=False):
    use_gpu=((not force_cpu) and torch.cuda.is_available())
    return torch.device(CUDA if use_gpu else CPU)