import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class atten(nn.Module):
    def __init__(self,method='mean'):
        super(atten,self).__init__()
        self.method=method
    def mean_method(self,hidden):
        temp=torch.sum(hidden,dim=1,keepdim=False)    #[bs,num_filters]
        l=temp.size()[-1]
        return temp/l   #[bs,num_filters]
    def forward(self,hidden):
        return self.mean_method(hidden)