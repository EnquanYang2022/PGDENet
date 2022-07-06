import torch
import torch as t
import torch.nn as nn

#用rgb增强depth
class BasicConv2d(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride=1,padding=0,dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class ER(nn.Module):
    def __init__(self, in_channel):
        super(ER, self).__init__()

        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 1, 1, bias=False),
                                     nn.BatchNorm2d(in_channel), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 4, 4, bias=False),
                                     nn.BatchNorm2d(in_channel), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 8, 8, bias=False),
                                     nn.BatchNorm2d(in_channel), nn.LeakyReLU(0.1, inplace=True))

        self.b_1 = BasicConv2d(in_channel * 3, in_channel, kernel_size=3, padding=1)
        self.conv_res = BasicConv2d(in_channel,in_channel,kernel_size=1,padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        buffer_1 = []
        buffer_1.append(self.conv1_1(x))
        buffer_1.append(self.conv2_1(x))
        buffer_1.append(self.conv3_1(x))
        buffer_1 = self.b_1(torch.cat(buffer_1, 1))
        out = self.relu(buffer_1+self.conv_res(x))

        return out
class FilterLayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(FilterLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y
class DA(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(DA, self).__init__()
        self.er = ER(inchannel)
        self.conv_r1 = nn.Conv2d(inchannel,inchannel,kernel_size=3,padding=1)
        self.sig = nn.Sigmoid()
        self.filter = FilterLayer(inchannel,outchannel)

        self.conv1 = BasicConv2d(in_channel=2*inchannel,out_channel=outchannel,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(outchannel,outchannel,kernel_size=1,padding=0)
        self.bn1 = nn.BatchNorm2d(outchannel)
    def forward(self,r,d):
        r1 = self.er(r)
        r1 = self.conv_r1(r1)
        r2 = self.filter(r)

        d1 = r2*d
        d1 = d1+d
        d1 = r1-d1
        d1 = self.sig(d1)
        d1 = r1*d1
        out = d+d1
        return out

