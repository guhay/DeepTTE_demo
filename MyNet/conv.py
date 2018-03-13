import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class conv_model(nn.Module):
    def __init__(self,kernel_size,num_filters):
        super(conv_model,self).__init__()
        self.kernel_size=kernel_size
        self.num_filters=num_filters
        self.build()
    def build(self):
        self.process_codes=nn.Linear(2,16)
        self.conv=nn.Conv1d(16,self.num_filters,self.kernel_size)
    def forward(self,traj):
        lngs=torch.unsqueeze(Variable(traj['lngs']),dim=-1)    #[bs,seq_length,1]
        lats=torch.unsqueeze(Variable(traj['lats']),dim=-1)    #[bs,seq_length,1]
        locs=torch.cat((lngs,lats),dim=2)   #[bs,seq_length,2]
        locs=F.tanh(self.process_codes(locs))   #[bs,seq_length,16]
        locs=locs.permute(0,2,1)
        conv_locs=F.elu(self.conv(locs)).permute(0,2,1) #[bs,conv(H),num_filters]
        return conv_locs