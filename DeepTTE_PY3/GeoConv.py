import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import numpy as np
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self,kernel_size,num_filter):
        super(Net,self).__init__()
        self.kernel_size=kernel_size
        self.num_filter=num_filter
        self.build()
    def build(self):
        self.state_em=nn.Embedding(2,2)
        self.process_cooeds=nn.Linear(4,16)
        self.conv=nn.Conv1d(16,self.num_filter,self.kernel_size)
    def forward(self,traj):
        lngs=torch.unsqueeze(Variable(traj['lngs']),dim=2)    #[bs,seq_length,1]
        lats=torch.unsqueeze(Variable(traj['lats']),dim=2)    #[bs,seq_length,1]
        states=self.state_em(Variable(traj['states']).long()) #[bs,seq_length,2]
        locs=torch.cat((lngs,lats,states),dim=2)    #[bs,seq_length,4]

        locs=F.tanh(self.process_cooeds(locs))  #[bs,seq_length,16]
        locs=locs.permute(0,2,1)    #[bs,16,seq_length]

        conv_locs=F.elu(self.conv(locs)).permute(0,2,1) #[bs,conv(H),num_filters]
        local_dist=utils.get_local_seq(Variable(traj['dist_gap']),self.kernel_size,'dist_gap_mean','dist_gap_std') #[bs,conv(H)]
        local_dist=torch.unsqueeze(local_dist,dim=2)    #[bs,conv(H),1]

        conv_locs=torch.cat((conv_locs,local_dist),dim=2)   #[bs,conv(H),num_filters+1]

        return conv_locs