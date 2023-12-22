import torch
import torch.nn as nn
import platform

def set_AwA2_dataset_path():
    system = platform.system()
    assert (system == 'Windows') | (system == 'Linux')
    if system == 'Windows':
        path = 'Animals_with_Attributes2/'
    elif system == 'Linux':
        path = '/gpu-data/pger/Animals_with_Attributes2/'
    return path









