import torch.nn as nn
import torch
from  torch.nn import functional as F

#相当于ASPP
# class ER(nn.Module):
#     def __init__(self, in_channel):
#         super(ER, self).__init__()
#
#         self.conv1_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 1, 1, bias=False),
#                                      nn.BatchNorm2d(in_channel), nn.ReLU())
#         self.conv2_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 4, 4, bias=False),
#                                      nn.BatchNorm2d(in_channel), nn.ReLU())
#         self.conv3_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 8, 8, bias=False),
#                                      nn.BatchNorm2d(in_channel), nn.ReLU())
#
#
#         self.glo = nn.AdaptiveAvgPool2d(1)
#
#     def forward(self, x):
#         x1 = self.conv1_1(x)
#         x2 = self.conv2_1(x1)
#         x2 = x2+x1
#         x3 = self.conv3_1(x2)
#         x3 = x3+x2
#         x4 = self.glo(x)
#         return x1,x2,x3,x4

# class ER(nn.Module):
#     def __init__(self, in_channel):
#         super(ER, self).__init__()
#
#         self.conv1_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 1, 1, bias=False),
#                                      nn.BatchNorm2d(in_channel), nn.LeakyReLU(0.1, inplace=True))
#         self.conv2_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 4, 4, bias=False),
#                                      nn.BatchNorm2d(in_channel), nn.LeakyReLU(0.1, inplace=True))
#         self.conv3_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 8, 8, bias=False),
#                                      nn.BatchNorm2d(in_channel), nn.LeakyReLU(0.1, inplace=True))
#
#         self.b_1 = BasicConv2d(in_channel * 3, in_channel, kernel_size=3, padding=1)
#         self.conv_res = BasicConv2d(in_channel,in_channel,kernel_size=1,padding=0)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         buffer_1 = []
#         buffer_1.append(self.conv1_1(x))
#         buffer_1.append(self.conv2_1(x))
#         buffer_1.append(self.conv3_1(x))
#         buffer_1 = self.b_1(torch.cat(buffer_1, 1))
#         out = self.relu(buffer_1+self.conv_res(x))
#
#         return out


class BN_Conv2d(nn.Module):
    """
    BN_CONV_RELU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False) -> object:
        super(BN_Conv2d, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return F.relu(self.seq(x))
class DenseBlock(nn.Module):

    def __init__(self, input_channels, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.k0 = input_channels
        self.k = growth_rate
        self.layers = self.__make_layers()

    def __make_layers(self):
        layer_list = []
        for i in range(self.num_layers):
            layer_list.append(nn.Sequential(
                BN_Conv2d(self.k0+i*self.k, 4*self.k, 1, 1, 0),
                BN_Conv2d(4 * self.k, self.k, 3, 1, 1)
            ))
        return layer_list

    def forward(self, x):
        feature = self.layers[0](x)
        out = torch.cat((x, feature), 1)
        for i in range(1, len(self.layers)):
            feature = self.layers[i](out)
            out = torch.cat((feature, out), 1)
        return out


class dense(nn.Module):
    def __init__(self,inchannel):
        super(dense, self).__init__()
        self.conv1 = BasicConv2d(inchannel, inchannel, kernel_size=3, padding=3, dilation=3)
        self.conv2 = BasicConv2d(inchannel, inchannel, kernel_size=3, padding=5, dilation=5)
        self.conv3 = BasicConv2d(inchannel, inchannel, kernel_size=3, padding=7, dilation=7)
    def forward(self,x,y):
        out = x+y
        c1 = self.conv1(out)

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
#解码中的sg(帅哥)
class SG(nn.Module):
    def __init__(self,inchannel,nextchannel):
        super(SG, self).__init__()


        self.conv_t = BasicConv2d(inchannel,nextchannel,kernel_size=1,padding=0)
        # self.conv0 = BasicConv2d(inchannel,inchannel,kernel_size=3,padding=1)
        # self.conv1 = BasicConv2d(inchannel,inchannel,kernel_size=3,padding=3,dilation=3)
        # self.conv2 = BasicConv2d(2*inchannel,inchannel,kernel_size=3,padding=5,dilation=5)
        # self.conv3 = BasicConv2d(3*inchannel, inchannel, kernel_size=3, padding=7, dilation=7)
        # self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.upsample = nn.Sequential(nn.ConvTranspose2d(inchannel, inchannel,kernel_size=3, stride=2, padding=1,output_padding=1, bias=False),
                                      nn.BatchNorm2d(inchannel),
                                      nn.ReLU(inplace=True)
                                      )

    def forward(self,x,pre_x):



        out = x+pre_x
        # out1 = self.conv0(o)
        # out2 = self.conv1(out1)
        # in2 = torch.cat((out1,out2),dim=1)
        # out3 = self.conv2(in2)
        # in3 = torch.cat((out1,out2,out3),dim=1)
        # out = self.conv3(in3)
        #
        # out = out+o

        final = self.upsample(out)
        final = self.conv_t(final)

        return final




# import torch.nn as nn
# import torch
# class TransBasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
#         super(TransBasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, inplanes,kernel_size=3,padding=1)
#         self.bn1 = nn.BatchNorm2d(inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.upsample = upsample
#         self.stride = stride
#         if upsample is not None and stride != 1:
#             self.conv2 = nn.ConvTranspose2d(inplanes, planes,
#                                             kernel_size=3, stride=stride, padding=1,
#                                             output_padding=1, bias=False)
#
#         else:
#             self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3,stride=stride,padding=1)
#         self.bn2 = nn.BatchNorm2d(planes)
#
#
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.upsample is not None:
#             residual = self.upsample(x)
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
#
# #相当于ASPP
# class ER(nn.Module):
#     def __init__(self, in_channel):
#         super(ER, self).__init__()
#
#         self.conv1_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 1, 1, bias=False),
#                                      nn.BatchNorm2d(in_channel), nn.LeakyReLU(0.1, inplace=True))
#         self.conv2_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 4, 4, bias=False),
#                                      nn.BatchNorm2d(in_channel), nn.LeakyReLU(0.1, inplace=True))
#         self.conv3_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 8, 8, bias=False),
#                                      nn.BatchNorm2d(in_channel), nn.LeakyReLU(0.1, inplace=True))
#
#         self.b_1 = BasicConv2d(in_channel * 3, in_channel, kernel_size=3, padding=1)
#         self.conv_res = BasicConv2d(in_channel,in_channel,kernel_size=1,padding=0)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#
#         buffer_1 = []
#         buffer_1.append(self.conv1_1(x))
#         buffer_1.append(self.conv2_1(x))
#         buffer_1.append(self.conv3_1(x))
#         buffer_1 = self.b_1(torch.cat(buffer_1, 1))
#         out = self.relu(buffer_1+self.conv_res(x))
#
#         return out
#
# class BasicConv2d(nn.Module):
#     def __init__(self,in_channel,out_channel,kernel_size,stride=1,padding=0,dilation=1):
#         super(BasicConv2d, self).__init__()
#         self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=False)
#         self.bn = nn.BatchNorm2d(out_channel)
#         self.relu = nn.ReLU(inplace=True)
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x
# #解码中的sg(帅哥)
# class SG(nn.Module):
#     def __init__(self,smallchannel,bigchannel,c=True,flag=None,in_plane=None):
#         super(SG, self).__init__()
#         self.c =c
#         self.flag = flag
#         #self.transconv = nn.ConvTranspose2d(bigchannel,smallchannel,kernel_size=1,padding=0)
#         #self.bn = nn.BatchNorm2d(smallchannel)
#         self.inplanes = in_plane
#         #self.conv = BasicConv2d(bigchannel,smallchannel,kernel_size=3,padding=1)
#         #self.conv_out = BasicConv2d(smallchannel,smallchannel,kernel_size=1,padding=0)
#         block = TransBasicBlock
#         self.er = ER(smallchannel)
#         self.trans1 = self._make_transpose(block,128,6,stride=2)
#         self.trans2 = self._make_transpose(block, 64, 4, stride=2)
#         self.trans3 = self._make_transpose(block, 64, 3, stride=2)
#         self.trans4 = self._make_transpose(block, 64, 3, stride=2)
#
#     def forward(self,x,pre_x):
#         # print(x.shape, pre_x.shape)
#         combine = x+pre_x
#         #combine = self.conv_out(combine)
#         combine = self.er(combine)
#         if self.flag==1:
#             combine = self.trans1(combine)
#         elif self.flag==2:
#             combine = self.trans2(combine)
#         elif self.flag==3:
#             combine = self.trans3(combine)
#         elif self.flag==4:
#             combine = self.trans4(combine)
#         return combine
#
#     def _make_transpose(self, block, planes, blocks, stride=1):
#         upsample = None
#         if stride != 1:
#             upsample = nn.Sequential(
#                 nn.ConvTranspose2d(self.inplanes, planes,
#                                    kernel_size=2, stride=stride,
#                                    padding=0, bias=False),
#                 nn.BatchNorm2d(planes),
#             )
#         elif self.inplanes != planes:
#             upsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes),
#             )
#
#         # upsample2 = nn.Sequential(
#         #     nn.Conv2d(self.inplanes, self.inplanes,
#         #               kernel_size=1, stride=1, bias=False),
#         #     nn.BatchNorm2d(self.inplanes),
#         # )
#
#         layers = []
#
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, self.inplanes))
#
#         layers.append(block(self.inplanes, planes, stride, upsample))
#         self.inplanes = planes
#
#         return nn.Sequential(*layers)