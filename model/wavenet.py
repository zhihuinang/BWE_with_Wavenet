import torch
import torch.nn as nn
import torch.nn.functional as F

class Aconv1d(nn.Module):
    def __init__(self,dilation,channel_in, channel_out, kernel_size=3):
        super(Aconv1d, self).__init__()


        self.dilation = dilation
        self.pad_num = int((kernel_size - 1) * dilation)
      
        self.dilation_conv1d = nn.Conv1d(in_channels=channel_in, out_channels=channel_out,
                                       kernel_size=kernel_size, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(channel_out)


    def forward(self, inputs):
        inputs = F.pad(inputs, (self.pad_num, 0))
        outputs = self.dilation_conv1d(inputs)
        outputs = self.bn(outputs)

        return outputs


class Condition_conv(nn.Module):
    # get the conditional input:Mel_spec of shape (B,1,h,w)
    # to the conditional h: of shape (B,C,1)
    def __init__(self,channel_in, channel_out):
        super(Condition_conv,self).__init__()
        self.conv_1 = nn.Conv2d(in_channels = channel_in, out_channels = channel_out,kernel_size=3)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=3)
        self.conv_2 = nn.Conv2d(in_channels = channel_out, out_channels = channel_out,kernel_size=3)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=3)
        self.conv_3 = nn.Conv2d(in_channels = channel_out, out_channels = channel_out,kernel_size=(6,1))
        
        
        
    def forward(self,input):
        input = input.unsqueeze(1)
        output = self.max_pool_1(self.conv_1(input))  # B,C,1
        output = self.max_pool_2(self.conv_2(output))
        output = self.conv_3(output)
        B, C, _, _ = output.shape  # B,256,1,1
        output = output.squeeze(-1)
        return output

class ResnetBlock(nn.Module):
    def __init__(self, dilation, channel_in, channel_out):
        super(ResnetBlock, self).__init__()
        
        self.conv_filter = Aconv1d(dilation, channel_in, channel_out)
        self.conv_gate = Aconv1d(dilation, channel_in, channel_out)
        self.condition_filter = Condition_conv(1,channel_out)
        self.condition_gate = Condition_conv(1,channel_out)
        self.conv1d = nn.Conv1d(channel_out, out_channels=channel_out, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channel_out)

    def forward(self, inputs,condition_input):
        B, _, T = inputs.shape

        h_filter = self.condition_filter(condition_input)
        h_gate = self.condition_gate(condition_input)
        h_filter = h_filter.expand(B,-1,T)
        h_gate = h_gate.expand(B,-1,T)

        out_filter = self.conv_filter(inputs)
        out_filter = out_filter + h_filter
        out_filter = F.tanh(out_filter)
        
        out_gate = self.conv_gate(inputs)
        out_gate = out_gate + h_gate
        out_gate = F.sigmoid(out_gate)
        
        outputs = out_filter * out_gate

        outputs = torch.tanh(self.bn(self.conv1d(outputs)))
        out = outputs + inputs
        return out, outputs

class WaveNet(nn.Module):
    def __init__(self, channels_in, channels_out=256, dilations=[1,2,4,8,16,32,64]):
        super(WaveNet, self).__init__()
        
        self.conv1d = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, kernel_size=3,padding=1,bias=False)
        self.bn = nn.BatchNorm1d(channels_out)

        self.resnet_block_0 = nn.ModuleList([ResnetBlock(dilation, channels_out, channels_out) for dilation in dilations])
        self.resnet_block_1 = nn.ModuleList([ResnetBlock(dilation, channels_out, channels_out) for dilation in dilations])
        self.resnet_block_2 = nn.ModuleList([ResnetBlock(dilation, channels_out, channels_out) for dilation in dilations])
        self.conv1d_out_0 = nn.Conv1d(channels_out, channels_out, kernel_size=1, bias=False)
        self.conv1d_out_1 = nn.Conv1d(channels_out, channels_out, kernel_size=1, bias=False)



    def forward(self, inputs,conditional_input):
        inputs = inputs.unsqueeze(1)
        x = self.bn(self.conv1d(inputs))
        x = torch.tanh(x)
        outs = 0.0
        for layer in self.resnet_block_0:
            x, out = layer(x,conditional_input)
            outs += out
        for layer in self.resnet_block_1:
            x, out = layer(x,conditional_input)
            outs += out
        for layer in self.resnet_block_2:
            x, out = layer(x,conditional_input)
            outs += out
        outs = F.relu(outs)
        outs = F.relu(self.conv1d_out_0(outs))
        outs = self.conv1d_out_1(outs)
        
        return outs