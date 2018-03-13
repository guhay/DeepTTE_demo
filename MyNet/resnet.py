import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class res_net(nn.Module):
    def __init__(self,num_hidden_size,input_size,hidden_size,output_size):
        super(res_net,self).__init__()
        self.input2hidden=nn.Linear(input_size,hidden_size)
        self.residuals=nn.ModuleList()
        for i in range(num_hidden_size):
            self.residuals.append(nn.Linear(hidden_size,hidden_size))
        self.hidden2output=nn.Linear(hidden_size,output_size)
    def forward(self,input_feature):
        hidden=self.input2hidden(input_feature)
        for i in range(len(self.residuals)):
            residual=F.relu(hidden)
            hidden=hidden+residual
        output=self.hidden2output(hidden)
        return output   #[bs,2]