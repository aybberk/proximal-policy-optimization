import torch

def discount(arr, coef):
    '''Comment'''
    length = arr.shape[-1]
    coefs = coef ** torch.arange(length)
    return coefs * arr
    
def flatten(arr):
    '''Comment'''
    return arr.reshape(-1, *arr.shape[2:])

def to_pytorch(state):
    state = state[:].transpose(2, 0, 1)
    state = torch.from_numpy(state)
    state.to("cuda")
    return state