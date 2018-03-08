import data_loader
from Attr import Attr_Net
from SpatioTemporal import Spatio_Net
from DeepTTE import DeepTTE_Net
from torch.autograd import Variable
import torch

from itertools import count

attr_net=Attr_Net()
spatio_net=Spatio_Net(28)
deepTTE_Net=DeepTTE_Net()

data_iter=data_loader.get_loader('train_00',5)
for idx, (attr, traj) in enumerate(data_iter):
    # attr_output=attr_net(attr)
    # packed_hiddens, lens,aaa,bbb=spatio_net(traj,attr_output)
    # print(packed_hiddens)
    # print(aaa)

    print(attr['time'])
    # entire_out, (local_out, sptm_l)=deepTTE_Net(attr,traj)
    # print(local_out)
    # print(sptm_l)
