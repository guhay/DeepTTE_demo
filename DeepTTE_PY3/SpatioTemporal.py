import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import GeoConv
import numpy as np

class Spatio_Net(nn.Module):
    def __init__(self,attr_size,kernel_size=3,num_filter=32,hidden_size=128,num_layers=2,pooling_method='mean',rnn='lstm'):
        super(Spatio_Net,self).__init__()

        self.kernel_size=kernel_size
        self.num_filter=num_filter
        self.pooling_method=pooling_method
        self.hidden_size=hidden_size
        self.geo_conv=GeoConv.Conv_Net(kernel_size,num_filter)

        if rnn=='lstm':
            self.rnn=nn.LSTM(input_size=self.num_filter+1+attr_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        else:
            self.rnn=nn.RNN(input_size=self.num_filter+1+attr_size,hidden_size=hidden_size,num_layers=2,batch_first=True)
    def mean_pooling(self,hiddens,lens):
        hiddens=torch.sum(hiddens,dim=1,keepdim=False)  #[bs,rnn_hidden_size]
        lens=torch.FloatTensor(lens)    #[bs]
        lens=Variable(torch.unsqueeze(lens,dim=1))  #[bs,1]
        return hiddens/lens
    def out_size(self):
        return int(self.hidden_size)
    def forward(self,traj,attr_t):
        conv_locs=self.geo_conv(traj)
        attr_t=torch.unsqueeze(attr_t,dim=1)    #[bs,1,attr_size]
        expand_attr_t=attr_t.expand(conv_locs.size()[:2]+(attr_t.size()[-1],))  #[bs,conv(H),attr_size]

        conv_locs=torch.cat((conv_locs,expand_attr_t),dim=2)    #[bs,conv(H),num_filter+1+attr_size]

        lens=list(map(lambda x:x-self.kernel_size+1,traj['lens']))

        packed_input=nn.utils.rnn.pack_padded_sequence(conv_locs,lens,batch_first=True)
        packed_hiddens,(h,c)=self.rnn(packed_input)
        hiddens,lens=nn.utils.rnn.pad_packed_sequence(packed_hiddens,batch_first=True)

        if self.pooling_method=='mean':
            return packed_hiddens,lens,self.mean_pooling(hiddens,lens)

