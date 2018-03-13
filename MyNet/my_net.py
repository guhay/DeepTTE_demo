import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from conv import conv_model
from attention import atten
from resnet import res_net

class Net(nn.Module):
    def __init__(self,kernel_size,num_filters):
        super(Net,self).__init__()
        self.kernel_size=kernel_size
        self.num_filters=num_filters
        self.conv=conv_model(self.kernel_size,self.num_filters)
        self.attention=atten()
        self.residuals=res_net(7,num_filters,50,2)
    def forward(self,traj):
        conv_locs=self.conv(traj)
        attention=self.attention(conv_locs)
        output=self.residuals(attention)
        return output
    def eval_on_batch(self,traj,label):
        output=self(traj)
        bs=output.size()[0]
        loss=torch.abs(output-label)/(bs*2)
        loss=torch.sum(loss)
        return loss