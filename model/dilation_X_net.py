import torch
import torch.nn as nn
import torch.nn.functional as F


def Swish(x):
    return x*F.sigmoid(x)

class Causalconv1d(nn.Module):
    def __init__(self,channel_in, channel_out, dilation,kernel_size,stride=1):
        super(Causalconv1d, self).__init__()


        self.dilation = dilation
        self.pad_num = int((kernel_size - 1) * dilation)
      
        self.dilation_conv1d = nn.Conv1d(in_channels=channel_in, out_channels=channel_out,
                                       kernel_size=kernel_size, dilation=dilation, stride=stride, bias=False)
   


    def forward(self, inputs):
        inputs = F.pad(inputs, (self.pad_num, 0))
        outputs = self.dilation_conv1d(inputs)

        return outputs


class Scale_Down(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Scale_Down,self).__init__()
        self.conv1 = Causalconv1d(in_channel,out_channel,1,4,stride=1)
        self.conv1_1 = Causalconv1d(out_channel,out_channel,2,4,1)
        self.conv1_2 = Causalconv1d(out_channel,out_channel,4,4,1)
        self.conv2 = Causalconv1d(out_channel,1,dilation=1,kernel_size=16,stride=2)

    def forward(self,input):
        output = self.conv1(input)
        output = self.conv1_1(output)
        output = self.conv1_2(output)
        output = F.tanh(self.conv2(output))
        return output

class Scale_Up(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Scale_Up,self).__init__()
        self.conv1 = Causalconv1d(in_channel,out_channel,1,4,stride=1)
        self.conv1_1 = Causalconv1d(out_channel,out_channel,2,4,1)
        self.conv1_2 = Causalconv1d(out_channel,out_channel,4,4,1)
        self.conv2 = Causalconv1d(out_channel,1,1,16,stride=1)
    def forward(self,input):
        output = Swish(self.conv1(input))
        output = Swish(self.conv1_1(output))
        output = Swish(self.conv1_2(output))
        output = F.tanh(self.conv2(output))
        return output

class Dilation_X_net(nn.Module):
    def __init__(self):
        super(Dilation_X_net,self).__init__()
        self.scale_down = Scale_Down(1,16)
        self.scale_up = Scale_Up(1,16)
    def forward(self,input,mode='down+up'):
        input = input.unsqueeze(1)
        if mode == 'down+up':
            Lo_res = self.scale_down(input)
            Lo_input = torch.repeat_interleave(Lo_res,2,dim=2)
            Hi_res = self.scale_up(Lo_input)
            Lo_res = Lo_res.squeeze()
            Hi_res = Hi_res.squeeze()
            return Lo_res, Hi_res
        elif mode == 'up':
            input = torch.repeat_interleave(input,2,dim=2)
            Hi_res = self.scale_up(input)
            Hi_res = Hi_res.squeeze()
            return Hi_res