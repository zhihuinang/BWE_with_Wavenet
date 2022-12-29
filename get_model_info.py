from thop import profile

from torchstat import stat
import torch
from model.X_net import *
from model.dilation_X_net import *

model1 = X_net()
model2 = Dilation_X_net()

input = torch.randn((2,5600))

flops, params = profile(model1, inputs=(input,))
print('X-net: Flops:{}, Params:{}'.format(flops,params))
flops, params = profile(model2, inputs=(input,))
print('Dilation X-net: Flops:{}, Params:{}'.format(flops,params))

