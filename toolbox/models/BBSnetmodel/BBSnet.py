import torch
import torch as t
import torch.nn as nn

from torch.autograd import Variable as V
import torchvision.models as models
from toolbox.models.BBSnetmodel.ResNet import ResNet50,ResNet34
from torch.nn import functional as F
from toolbox.models.BBSnetmodel.fusion import yfusion,yfusion_layer4
from toolbox.models.BBSnetmodel.DEM import DA
from toolbox.models.BBSnetmodel.SG import SG
from toolbox.models.BBSnetmodel.ASPP import ASPP
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
#         return x*aux






# BBSNet
class BBSNet(nn.Module):
    def __init__(self, channel=32,n_class=None):
        super(BBSNet, self).__init__()

        # Backbone model

        self.resnet = ResNet34('rgb')
        self.resnet_depth = ResNet34('rgbd')

        #DA rgb->depth
        self.da1 = DA(64,64)
        self.da2 = DA(64,64)
        self.da3 = DA(128,128)
        self.da4 = DA(256,256)
        self.da5 = DA(512,512)
        #DA depth->rgb
        # self.da1_r = DA(64, 64)
        # self.da2_r = DA(64, 64)
        # self.da3_r = DA(128, 128)
        # self.da4_r = DA(256, 256)
        # self.da5_r = DA(512, 512)

        #融合
        self.fusions = nn.ModuleList([
            yfusion(pre_inchannel=128,inchannel=64),
            yfusion(pre_inchannel=128,inchannel=64),
            yfusion(pre_inchannel=256,inchannel=128),
            yfusion(pre_inchannel=512,inchannel=256)
        ])

        self.fusion_layer4 = yfusion_layer4(512)



#layer1_fusion细化conv1
        self.conv1 = nn.Conv2d(64,128,kernel_size=3,padding=1)

        self.conv_end = nn.Conv2d(64,n_class,kernel_size=1,padding=0)

        self.sgs = nn.ModuleList([
            SG(256,128),
            SG(128,64),
            SG(64,64),
            SG(64,64)
        ])

        # self.sgs = nn.ModuleList([
        #     SG(256, 512, flag=1, in_plane=256),
        #     SG(128, 256, flag=2, in_plane=128),
        #     SG(64, 128, flag=3, in_plane=64),
        #     SG(64, 64, c=False, flag=4, in_plane=64)
        # ])
        # self.aspp = ASPP(num_classes=n_class)
        #处理layer4_fusion
        self.transconv = nn.ConvTranspose2d(512, 256, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(256)

        #辅助损失
        self.conv_aux1 = nn.Conv2d(128,n_class,kernel_size=1,stride=1)
        self.conv_aux2 = nn.Conv2d(64, n_class, kernel_size=1, stride=1)
        self.conv_aux3 = nn.Conv2d(64, n_class, kernel_size=1, stride=1)
        self.conv_aux4 = nn.Conv2d(64, n_class, kernel_size=1, stride=1)

        #加载预训练
        if self.training:
            self.initialize_weights()

    def forward(self, x, x_depth):

        x_depth=x_depth[:,:1,...]

        #conv1  64 ,1/4  ,此层暂不做融合
        x1 = self.resnet.conv1(x)
        x1 = self.resnet.bn1(x1)
        x1 = self.resnet.relu(x1)


        #h,w = x1.size()[2:]
        x_depth1 = self.resnet_depth.conv1(x_depth)
        x_depth1 = self.resnet_depth.bn1(x_depth1)
        x_depth1 = self.resnet_depth.relu(x_depth1)
        x_depth_da_conv1 = self.da1(x1, x_depth1)




        #layer1  64 1/4
        x2 = self.resnet.maxpool(x1)
        x2 = self.resnet.layer1(x2)
        x_depth2 = self.resnet_depth.maxpool(x_depth1)
        x_depth2 = self.resnet_depth.layer1(x_depth2)

        x_depth_da_layer1 = self.da2(x2,x_depth2)


        #layer2  128  1/8
        x3 = self.resnet.layer2(x2)
        x_depth3 = self.resnet_depth.layer2(x_depth2)

        x_depth_da_layer2 = self.da3(x3,x_depth3)


        #layer3 256 1/16
        x4 = self.resnet.layer3_1(x3)
        x_depth4 = self.resnet_depth.layer3_1(x_depth3)

        x_depth_da_layer3 = self.da4(x4,x_depth4)


        #layer4 512 1/32
        x5 = self.resnet.layer4_1(x4)
        x_depth5 = self.resnet_depth.layer4_1(x_depth4)

        x_depth_da_layer4 = self.da5(x5,x_depth5)


        #layer4的融和

        layer4_fusion = self.fusion_layer4(x5,x_depth_da_layer4)

        #layer3的融合

        layer3_fusion = self.fusions[3](x4,x_depth_da_layer3,layer4_fusion)

        #layer2的融合

        layer2_fusion = self.fusions[2](x3,x_depth_da_layer2,layer3_fusion)


        #layer1的融合

        layer1_fusion = self.fusions[1](x2,x_depth_da_layer1,layer2_fusion)

        #conv1的融合
        #x1 = x1+self.conv5_1(self.upsample8(x5))+self.conv2_1(x2)
        pre_layer1 = self.conv1(layer1_fusion)
        # pre_layer1 = self.upsample2(pre_layer1)
        conv1_fusion = self.fusions[0](x1,x_depth_da_conv1,pre_layer1)

        layer4_fusion = self.transconv(layer4_fusion)
        layer4_fusion = self.bn(layer4_fusion)
        h,w = layer3_fusion.size()[2:]
        layer4_fusion = F.interpolate(layer4_fusion,size=(h,w),mode='bilinear',align_corners=True)
        out1 = self.sgs[0](layer4_fusion,layer3_fusion)

        out2 = self.sgs[1](out1,layer2_fusion)

        out3 = self.sgs[2](out2,layer1_fusion)

        out4 = self.sgs[3](out3,conv1_fusion)

        out = self.conv_end(out4)
        out1_aux = self.conv_aux1(out1)
        out2_aux = self.conv_aux2(out2)
        out3_aux = self.conv_aux3(out3)

        if self.training:
            return out1_aux,out2_aux,out3_aux,out


        else:
            return out









    # initialize the weights
    def initialize_weights(self):

        #pretrain_dict = model_zoo.load_url(model_urls['resnet50'])
        res34 = models.resnet34(pretrained=True)
        pretrained_dict = res34.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)

        all_params = {}
        for k, v in self.resnet_depth.state_dict().items():
            if k == 'conv1.weight':
                all_params[k] = torch.nn.init.normal_(v, mean=0, std=1)
            elif k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet_depth.state_dict().keys())
        self.resnet_depth.load_state_dict(all_params)

if __name__ == '__main__':
    # x = V(t.randn(2,3,480,640))
    # y = V(t.randn(2,3,480,640))
    net = BBSNet(n_class=41)
    # print(list(net.parameters())[0])
    # print(net.named_parameters())
    # net1,net2,net3,net4= net(x,y)
    # print(net1.shape)
    # print(net2.shape)
    # print(net3.shape)
    # print(net4.shape)

    # from torchsummary import summary
    # model = BBSNet(n_class=41)
    # model = model.cuda()
    # summary(model, input_size=[(3, 480, 640),(3,480,640)],batch_size=6)

    print(sum(p.numel() for p in net.parameters())/1000000.0)