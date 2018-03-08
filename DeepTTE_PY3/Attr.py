import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import utils

class Attr_Net(nn.Module):
    embeded_dims=[('driverID',24000,16),('weekID',7,3),('timeID',1440,8)]

    def __init__(self):
        super(Attr_Net,self).__init__()
        self.build()
    def build(self):
        for name,dim_in,dim_out in Attr_Net.embeded_dims:
            self.add_module(name+'_em',nn.Embedding(dim_in,dim_out))
    def out_size(self):
        sz=0
        for name,dim_in,dim_out in Attr_Net.embeded_dims:
            sz+=dim_out
        return int(sz+1)
    def forward(self,attr):
        em_list=[]
        for name,dim_in,dim_out in Attr_Net.embeded_dims:
            embed=getattr(self,name+'_em')
            attr_t=embed(Variable(attr[name].view(-1,1))) #[bs,1,dim_out]
            attr_t=torch.squeeze(attr_t)    #[bs,dim_out]
            em_list.append(attr_t)
        dist=utils.normalize(Variable(attr['dist']),'dist')   #[bs]
        dist=dist.view(-1,1)    #[bs,1]
        em_list.append(dist)
        return torch.cat(em_list,dim=1) #[bs,sum(dim_out)+1]