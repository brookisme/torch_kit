import torch
import numpy as np
#
# CONFIG
#
CUDA='cuda'
CPU='cpu'


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