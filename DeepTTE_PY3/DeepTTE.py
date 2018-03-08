import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Attr import Attr_Net
from SpatioTemporal import Spatio_Net
from GeoConv import Conv_Net
import utils

EPS=10
class EntireEstimator(nn.Module):
    def __init__(self,input_size,num_fimal_fcs,hidden_size=128):
        super(EntireEstimator,self).__init__()
        self.input2hidden=nn.Linear(input_size,hidden_size)
        self.residuals=nn.ModuleList()
        for i in range(num_fimal_fcs):
            self.residuals.append(nn.Linear(hidden_size,hidden_size))
        self.hidden2out=nn.Linear(hidden_size,1)

    def forward(self,attr_t,sptm_t):
        inputs=torch.cat((attr_t,sptm_t),dim=1)
        hidden=self.input2hidden(inputs)
        for i in range(len(self.residuals)):
            residual=F.relu(self.residuals[i](hidden))
            hidden=hidden+residual
        out=self.hidden2out(hidden)
        return out  #[bs,1]

    def eval_on_batch(self,pred,label,mean,std):
        label=label.view(-1,1)
        pred=pred*std+mean
        label=label*std+mean
        loss=torch.abs(pred-label)/label
        return {'label':label,'pred':pred},loss.mean()

class LocalEstimator(nn.Module):
    def __init__(self,input_size):
        super(LocalEstimator,self).__init__()
        self.input2hidden=nn.Linear(input_size,64)
        self.hidden2hidden=nn.Linear(64,32)
        self.hidden2out=nn.Linear(32,1)
    def forward(self,sptm_s):
        hidden=F.leaky_relu(self.input2hidden(sptm_s))
        hidden=F.leaky_relu(self.hidden2hidden(hidden))
        out=self.hidden2out(hidden)
        return out  #[bs*every_length,1]
    def eval_on_batch(self,pred,lens,label,mean,std):
        label=nn.utils.rnn.pack_padded_sequence(label,lens,batch_first=True)
        label=label.view(-1,1)  #[bs*conv(H),1]

        label=label*std+mean
        pred=pred*std+mean
        loss=torch.abs(pred-label)/(label+EPS)

        return loss.mean()
class DeepTTE_Net(nn.Module):
    def __init__(self,kernel_size=3,num_filter=32,pooling_meathod='mean',num_final_fcs=3,final_fc_size=128,alpha=0.3):
        super(DeepTTE_Net,self).__init__()
        self.kernel_size=kernel_size
        self.num_filter=num_filter
        self.pooling_meathod=pooling_meathod
        self.num_final_fcs=num_final_fcs
        self.final_fc_size=final_fc_size
        self.alpha=alpha
        self.build()
        self.init_weight()
    def build(self):
        self.attr_net=Attr_Net()
        self.geo_conv=Conv_Net(self.kernel_size,self.num_filter)
        self.spatio_net=Spatio_Net(
            attr_size=self.attr_net.out_size(),
            kernel_size=self.kernel_size,
            num_filter=self.num_filter,
        )
        input_size=self.spatio_net.out_size()+self.attr_net.out_size()
        self.entire_estimate=EntireEstimator(
            input_size=input_size,
            num_fimal_fcs=self.num_final_fcs,
            hidden_size=self.final_fc_size,
        )

        self.loacl_estimate=LocalEstimator(input_size=128)
    def init_weight(self):
        for name,param in self.named_parameters():
            if name.find('.bias') != -1:
                param.data.fill_(0)
            elif name.find('.weight')!=-1:
                nn.init.xavier_uniform(param.data)
    def forward(self,attr,traj):
        attr_t=self.attr_net(attr)
        sptm_s,sptm_l,sptm_t=self.spatio_net(traj,attr_t)
        entire_out=self.entire_estimate(attr_t,sptm_t)

        if self.training:
            local_out=self.loacl_estimate(sptm_s[0])    #[bs*every_length,1]
            return entire_out,(local_out,sptm_l)    #[bs,1] [bs*every_conv(H),1] [bs]
        else:
            return entire_out

    def eval_on_batch(self,attr,traj,config):
        if self.training:
            entire_out,(local_out,local_length)=self(attr,traj)
        else:
            entire_out=self(attr,traj)
        pred_dict,entire_loss=self.entire_estimate.eval_on_batch(entire_out,attr['time'],config['time_mean'],config['time_std'])    #entire_loss is scalar

        if self.training:
            mean,std=(self.kernel_size-1)*config['time_gap_mean'],(self.kernel_size-1)*config['time_gap_std']

            local_label=utils.get_local_seq(traj['time_gap'],self.kernel_size,mean,std)
            local_loss=self.loacl_estimate.eval_on_batch(local_out,local_length,local_label,mean,std)

            return pred_dict,(1-self.alpha)*entire_loss+self.alpha*local_loss
        else:
            return pred_dict,entire_loss

