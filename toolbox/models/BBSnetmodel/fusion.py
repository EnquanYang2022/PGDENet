import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
#1,2,3,4层

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

# class denseblock(nn.Module):
#     def __init__(self,inchannel):
#         super(denseblock, self).__init__()
#         self.conv1 = BasicConv2d(inchannel,64,kernel_size=1,padding=0)
#         self.conv2 = BasicConv2d(64+inchannel, 64, kernel_size=3, padding=1)
#         self.conv3 = BasicConv2d(inchannel+64*2, 64, kernel_size=3, padding=1)
#         self.conv4 = BasicConv2d(inchannel+64*3, inchannel, kernel_size=3, padding=1)
#     def forward(self,input):
#         o1 = self.conv1(input)
#         o2_in = torch.cat((input,o1),dim=1)
#         o2 = self.conv2(o2_in)
#         o3_in = torch.cat((input,o1,o2),dim=1)
#
#         o3 = self.conv3(o3_in)
#         o4_in = torch.cat((input,o1,o2,o3),dim=1)
#         o4 = self.conv4(o4_in)
#         return o4




# class FilterLayer(nn.Module):
#     def __init__(self, in_planes, out_planes, reduction=16):
#         super(FilterLayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_planes, out_planes // reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(out_planes // reduction, out_planes),
#             nn.Sigmoid()
#         )
#         self.out_planes = out_planes
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, self.out_planes, 1, 1)
#         return y
#ACM
# class acm(nn.Module):
#     def __init__(self,num_channel):
#
#         super(acm, self).__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv2d(num_channel,num_channel,kernel_size=1)
#         self.activation = nn.Sigmoid()
#     def forward(self,x):
#         aux = self.pool(x)
#         aux = self.conv(aux)
#         aux = self.activation(aux)
#         return x*aux-
#GCM
class GCM(nn.Module):
    def __init__(self,inchannels):
        super(GCM, self).__init__()
        self.branches0 = nn.Sequential(
            BasicConv2d(inchannels,inchannels,kernel_size=1,padding=0)
        )
        self.branches1 = nn.Sequential(

            BasicConv2d(inchannels,inchannels,kernel_size=1,padding=0),

            BasicConv2d(inchannels,inchannels,kernel_size=3,padding=1,dilation=1)
        )
        self.branches2 = nn.Sequential(
            BasicConv2d(inchannels, inchannels, kernel_size=1,padding=0),

            BasicConv2d(inchannels, inchannels, kernel_size=5, padding=2, dilation=1)
        )
        self.branches3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,padding=1,stride=1),
            BasicConv2d(inchannels, inchannels, kernel_size=1,padding=0),

            # BasicConv2d(inchannels, inchannels, kernel_size=3, padding=7, dilation=7)
        )
        self.conv1 = BasicConv2d(4*inchannels,inchannels,kernel_size=3,padding=1)
        # self.conv2 = BasicConv2d(inchannels,outchannels,kernel_size=1)
    def forward(self,x):
        x0 = self.branches0(x)
        x1 = self.branches1(x)
        x2 = self.branches2(x)
        x3 = self.branches3(x)
        # print(x0.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        out_cat = self.conv1(torch.cat((x0,x1,x2,x3),dim=1))
        # out = self.conv1(out_cat)
        # out = out_cat+out_x
        return out_cat
class yfusion(nn.Module):
    def __init__(self,pre_inchannel,inchannel):
        super(yfusion, self).__init__()
        self.conv13 = BasicConv2d(in_channel=inchannel, out_channel=inchannel, kernel_size=(1, 3), padding=(0, 1))
        self.conv31 = BasicConv2d(in_channel=inchannel, out_channel=inchannel, kernel_size=(3, 1), padding=(1, 0))

        self.conv13_2 = BasicConv2d(in_channel=inchannel, out_channel=inchannel, kernel_size=(1, 3), padding=(0, 1))
        self.conv31_2 = BasicConv2d(in_channel=inchannel, out_channel=inchannel, kernel_size=(3, 1), padding=(1, 0))
        self.conv_w = nn.Conv2d(inchannel, 1, kernel_size=3, padding=1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        # self.activation = nn.Sigmoid()
        self.conv_aux = nn.Conv2d(2 * inchannel, inchannel, kernel_size=1, padding=0)
        self.conv_g = nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1)
        self.conv_g2 = nn.Conv2d(inchannel,inchannel,kernel_size=3,padding=1)
        self.sig = nn.Sigmoid()

        # self.conv_aux2 = nn.Conv2d(2*inchannel, inchannel, kernel_size=1)
        self.conv_m = nn.Conv2d(2*inchannel,inchannel,kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv_pre = nn.Conv2d(pre_inchannel,inchannel,kernel_size=3,padding=1)
        self.gcm = GCM(inchannel)

    def forward(self,rgb,depth,pre):
        dense_r = self.conv31(self.conv13(rgb))

        dense_d = self.conv13_2(self.conv31_2(depth))
        weight = self.conv_w(dense_r * dense_d)
        dense_r = weight + dense_r
        dense_d = weight + dense_d
        m = torch.cat((dense_r, dense_d), dim=1)

        aux = torch.cat((rgb, depth), dim=1)
        aux = self.conv_aux(aux)
        g = self.conv_g(aux)
        g = self.sig(g)
        pre = self.conv_pre(pre)
        pre1 = self.upsample(pre)

        g = g*pre1

        aux = aux+g

        g2 = self.conv_g2(aux)
        g2 = self.sig(g2)
        pre2 = self.pool(pre)
        g2 = g2*pre2

        aux = aux+g2


        m = self.conv_m(m)
        m = m + aux
        # m_a = self.pool(m)
        # m_a = self.activation(m_a)
        # m = m_a * m
        m = self.gcm(m)
        return m



#特殊情况为第一层，没有pre_rgb输入
class yfusion_layer4(nn.Module):
    def __init__(self, inchannel):
        super(yfusion_layer4, self).__init__()
        self.conv13 = BasicConv2d(in_channel=inchannel, out_channel=inchannel, kernel_size=(1, 3), padding=(0, 1))
        self.conv31 = BasicConv2d(in_channel=inchannel, out_channel=inchannel, kernel_size=(3, 1), padding=(1, 0))

        self.conv13_2 = BasicConv2d(in_channel=inchannel, out_channel=inchannel, kernel_size=(1, 3), padding=(0, 1))
        self.conv31_2 = BasicConv2d(in_channel=inchannel, out_channel=inchannel, kernel_size=(3, 1), padding=(1, 0))
        self.conv_w = nn.Conv2d(inchannel,1,kernel_size=3,padding=1)
        self.conv_e = nn.Conv2d(2*inchannel,inchannel,kernel_size=1,padding=0)
        # self.pool = nn.AdaptiveAvgPool2d(1)
        # self.conv = nn.Conv2d(inchannel, inchannel, kernel_size=1)
        # self.activation = nn.Sigmoid()
        self.gcm = GCM(inchannel)

    def forward(self, rgb, depth):
        dense_r = self.conv31(self.conv13(rgb))

        dense_d = self.conv13_2(self.conv31_2(depth))
        weight = self.conv_w(dense_r*dense_d)
        dense_r = weight+dense_r
        dense_d = weight+dense_d
        m = torch.cat((dense_r,dense_d),dim=1)
        m = self.conv_e(m)
        # m_a = self.pool(m)
        # m_a = self.activation(m_a)
        # m = m_a*m
        m = self.gcm(m)
        return m


if __name__ == '__main__':
    fusion = yfusion(128,64)
    x = torch.randn(2,64,480,640)
    y = torch.randn(2,64,480,640)
    pre = torch.randn(2,128,240,320)
    out = fusion(x,y,pre)
    # print(x)
    # sof = nn.Softmax(dim=1)
    # out = sof(x)
    # print(out)
    print(out.shape)




