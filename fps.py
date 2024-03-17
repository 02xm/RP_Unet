import time
import torch
import numpy as np
from archs import *

Device = torch.device('cpu')

net=NestedUNet(1, 3)
net.eval()

# x是输入图片的大小
x = torch.zeros((4,3,224,112)).to(Device)
t_all = []

for i in range(100):
    t1 = time.time()
    y = net(x)
    t2 = time.time()
    t_all.append(t2 - t1)

print('average time:', np.mean(t_all) / 1)
print('average fps:',1 / np.mean(t_all))

print('fastest time:', min(t_all) / 1)
print('fastest fps:',1 / min(t_all))

print('slowest time:', max(t_all) / 1)
print('slowest fps:',1 / max(t_all))
