import torch
from torch.autograd import Variable
import json

config=json.load(open('./config.json','r'))

def normalize(x,key):
    mean=config[key+'_mean']
    std=config[key+'_std']
    return (x-mean)/std
def get_local_seq(full_seq,kernel_size,mean_str,std_str):
    mean=config[mean_str]
    std=config[std_str]
    seq_len=full_seq.size()[1]
    indices=Variable(torch.arange(0,seq_len).long())
    first_seq=torch.index_select(full_seq,dim=1,index=indices[kernel_size-1:])
    second_seq=torch.index_select(full_seq,dim=1,index=indices[:-kernel_size+1])
    local_seq=first_seq-second_seq
    local_seq=(local_seq-mean)/std
    return local_seq
