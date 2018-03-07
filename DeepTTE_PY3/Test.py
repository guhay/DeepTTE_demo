import data_loader
from Attr import Attr_Net
from SpatioTemporal import Spatio_Net
from torch.autograd import Variable
import torch

from itertools import count

attr_net=Attr_Net()
spatio_net=Spatio_Net(28)
data_iter=data_loader.get_loader('train_00',5)
for idx, (attr, traj) in enumerate(data_iter):
    attr_output=attr_net(attr)
    packed_hiddens, lens,aaa=spatio_net(traj,attr_output)
    print(aaa)

