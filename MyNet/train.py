from my_net import Net
import data_loader
import torch
from torch.autograd import Variable
import torch.optim as optim
import json

config=json.load(open('./config.json','r'))
def get_label(traj):
    lngs_last=torch.unsqueeze(Variable(traj['lngs_last']),dim=-1)
    lats_last=torch.unsqueeze(Variable(traj['lats_last']),dim=-1)
    return torch.cat((lngs_last,lats_last),dim=-1)  #[bs,2]
model=Net(3,10)
loss=torch.nn.MSELoss(size_average=True)
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
for i in range(10):
    for i in range(5):
        data_iter = data_loader.get_loader('train_0'+str(i), 5)
        for idx, (attr, traj) in enumerate(data_iter):
            label=get_label(traj)
            output=model(traj)
            l=loss(output,label)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            temp_std=Variable(torch.FloatTensor([0.04988770679679998, 0.04154695076189434]).expand_as(output))
            temp_mean=Variable(torch.FloatTensor([104.05810954320589,30.652312982784895]))
            print(output*temp_std+temp_mean)
            print(label*temp_std+temp_mean)