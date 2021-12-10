'''
Author: your name
Date: 2020-11-21 20:30:33
LastEditTime: 2020-11-22 01:23:21
LastEditors: Liu Chen
Description: 
FilePath: \BACN\networks.py
  
'''
import torch
import torchvision
from torch.nn.modules import LSTM
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import copy
import numpy as np
from matplotlib import pyplot as plt


class CNN_Layer(nn.Module):
    """
    bn: if Ture, use batch normalization
    pooling: tuple - (type, kernel_size, strides, padding)
                type: 'max', 'avg';
                kernel_size, strides: int or tuple
                padding: int
                pooling=None: don not use pooling layer
    act: choose an activation: 'relu' ... , None do not use activation
    """

    def __init__(self, inc, outc,
                 kernel, strides=1, padding=0,
                 bn=True, pooling=None, act='relu'):

        super(CNN_Layer, self).__init__()
        activations = {'relu': nn.ReLU}
        pool_methods = {'max': nn.MaxPool2d}

        cnn = []
        cnn.append(nn.Conv2d(inc, outc, kernel, strides, padding))
        if bn is True:
            cnn.append(nn.BatchNorm2d(outc))
        if act is not None:
            cnn.append(activations[act]())
        if pooling is not None:
            cnn.append(pool_methods[pooling[0]](pooling[1],
                                                pooling[2], pooling[3]))

        CNN_BN_ReLu = nn.Sequential(*cnn)

        self.cnn_layer = CNN_BN_ReLu

    def forward(self, x):
        return self.cnn_layer(x)
import torch.nn as nn

eps_fea_norm = 1e-5
eps_l2_norm = 1e-10

class SOSNet(nn.Module):
    """
       128-dimensional SOSNet model definition trained on 32x32 patches
       ：
       self.layers = nn.Sequential(
            nn.InstanceNorm2d(1, affine=False, eps=eps_fea_norm),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32, affine=False, eps=eps_fea_norm),
            activation,

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(64, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, affine=False, eps=eps_fea_norm),
            activation,

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
            activation,

            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
        )
       """
    def __init__(self):
        self.name = "MatchNet"
        super(MatchNet, self).__init__()
        FCs = [nn.Linear(512, 2048),
               nn.ReLU(),
               nn.Linear(2048, 512),
               nn.ReLU(),
               nn.Linear(512, 2)]
        self.metric_network = nn.Sequential(*FCs)
        self.BottleNeck = nn.Linear(256, 256)
        CNNs = []
        CNNs.append(CNN_Layer(3, 24, 7, padding=3, pooling=('max', 2, 2, 0)))
        CNNs.append(CNN_Layer(24, 64, 5, padding=2, pooling=('max', 3, 2, 0)))
        CNNs.append(CNN_Layer(64, 96, 3, padding=0))
        CNNs.append(CNN_Layer(96, 96, 1, padding=0))
        CNNs.append(CNN_Layer(96, 64, 3, strides=2,
                              padding=0, pooling=('max', 3, 2, 0)))
        self.FeatureNetwork = nn.Sequential(*CNNs)

    def forward(self, xa, xb):
        fa = self.FeatureNetwork(xa)
        fa = torch.flatten(fa, start_dim=1)
        fa = self.BottleNeck(fa)
        fb = self.FeatureNetwork(xb)
        fb = torch.flatten(fb, start_dim=1)
        fb = self.BottleNeck(fb)
        f = torch.cat([fa, fb], 1)
        out = self.metric_network(f)
        # out = f / f.detach().pow(2).sum(1, keepdim=True).sqrt()

        return out

class SOLAR(nn.Module):
        """
           128-dimensional solar model definition trained on 32x32 patches
            bacbone=matchnet
            loss:from https://github.com/tonyngjichun/SOLAR
            sos + fos

           """

        def __init__(self):
            self.name = "MatchNet"
            super(MatchNet, self).__init__()
            FCs = [nn.Linear(512, 2048),
                   nn.ReLU(),
                   nn.Linear(2048, 512),
                   nn.ReLU(),
                   nn.Linear(512, 2)]
            self.metric_network = nn.Sequential(*FCs)
            self.BottleNeck = nn.Linear(256, 256)
            CNNs = []
            CNNs.append(CNN_Layer(3, 24, 7, padding=3, pooling=('max', 2, 2, 0)))
            CNNs.append(CNN_Layer(24, 64, 5, padding=2, pooling=('max', 3, 2, 0)))
            CNNs.append(CNN_Layer(64, 96, 3, padding=0))
            CNNs.append(CNN_Layer(96, 96, 1, padding=0))
            CNNs.append(CNN_Layer(96, 64, 3, strides=2,
                                  padding=0, pooling=('max', 3, 2, 0)))
            self.FeatureNetwork = nn.Sequential(*CNNs)

        def forward(self, xa, xb):
            fa = self.FeatureNetwork(xa)
            fa = torch.flatten(fa, start_dim=1)
            fa = self.BottleNeck(fa)
            fb = self.FeatureNetwork(xb)
            fb = torch.flatten(fb, start_dim=1)
            fb = self.BottleNeck(fb)
            f = torch.cat([fa, fb], 1)
            out = self.metric_network(f)
            # out = f / f.detach().pow(2).sum(1, keepdim=True).sqrt()
            #loss: from https://github.com/tonyngjichun/SOLAR
            #sos + fos
            return out

class MatchNet(nn.Module):
    """
    OutPut: stacked features: shape [2, batchsize, feat_dim],
             A_feats and B_feats are in it
    """

    def __init__(self):
        self.name = "MatchNet"
        super(MatchNet, self).__init__()
        FCs = [nn.Linear(512, 2048),
               nn.ReLU(),
               nn.Linear(2048, 512),
               nn.ReLU(),
               nn.Linear(512,2)]
        self.metric_network = nn.Sequential(*FCs)
        self.BottleNeck = nn.Linear(256, 256)
        CNNs = []
        CNNs.append(CNN_Layer(3, 24, 7, padding=3, pooling=('max', 2, 2, 0)))
        CNNs.append(CNN_Layer(24, 64, 5, padding=2, pooling=('max', 3, 2, 0)))
        CNNs.append(CNN_Layer(64, 96, 3, padding=0))
        CNNs.append(CNN_Layer(96, 96, 1, padding=0))
        CNNs.append(CNN_Layer(96, 64, 3, strides=2,
                              padding=0, pooling=('max', 3, 2, 0)))
        self.FeatureNetwork = nn.Sequential(*CNNs)

    def forward(self, xa, xb):
        fa = self.FeatureNetwork(xa)
        fa = torch.flatten(fa, start_dim=1)
        fa = self.BottleNeck(fa)
        fb = self.FeatureNetwork(xb)
        fb = torch.flatten(fb, start_dim=1)
        fb = self.BottleNeck(fb)
        f = torch.cat([fa,fb],1)
        out = self.metric_network(f)
        # out = f / f.detach().pow(2).sum(1, keepdim=True).sqrt()

        return out


class Baseline(nn.Module):
    # 输入必须是 64x64
    def __init__(self):
        self.name = "Baseline"
        super(Baseline, self).__init__()
        # A_bot = [CNN_Layer(3, 32, 3, padding=1),
        #          CNN_Layer(32, 3, 3, padding=1)]
        # B_bot = A_bot.copy()
        # self.CNNA = nn.Sequential(*A_bot)
        # self.CNNB = nn.Sequential(*B_bot)
        CNNs = []
        CNNs.append(CNN_Layer(6, 24, 7, padding=3, pooling=('max', 2, 2, 0)))
        CNNs.append(CNN_Layer(24, 64, 5, padding=2, pooling=('max', 3, 2, 0)))
        CNNs.append(CNN_Layer(64, 96, 3, padding=0))
        CNNs.append(CNN_Layer(96, 96, 1, padding=0))
        CNNs.append(CNN_Layer(96, 64, 3, strides=2,
                              padding=0, pooling=('max', 3, 2, 0)))
        self.FeatureNetwork = nn.Sequential(*CNNs)
        FCs = [nn.Linear(256, 1024),
               nn.ReLU(),
               nn.Linear(1024, 2),
               nn.PReLU()
               ]
        self.fc = nn.Sequential(*FCs)

    def forward(self, xa, xb):
        # a = self.CNNA(xa)
        # b = self.CNNB(xb)
        x = torch.cat([xa, xb], dim=1)
        f = self.FeatureNetwork(x)
        f = torch.flatten(f, start_dim=1)
        out = self.fc(f)
        return out

# matchnet的CNN层但是在结尾增加了两个fc层映射到2 channels，然后用交叉熵训练分类
# 用以和合并通道的BACN的CNN层结构进行对比


class SplitNet(nn.Module):
    def __init__(self):
        self.name = "SplitNet"
        super(SplitNet, self).__init__()

        CNNs = []
        CNNs.append(CNN_Layer(3, 24, 7, padding=3, pooling=('max', 2, 2, 0)))
        CNNs.append(CNN_Layer(24, 64, 5, padding=2, pooling=('max', 3, 2, 0)))
        CNNs.append(CNN_Layer(64, 96, 3, padding=0))
        CNNs.append(CNN_Layer(96, 96, 1, padding=0))
        CNNs.append(CNN_Layer(96, 64, 3, strides=2,
                              padding=0, pooling=('max', 3, 2, 0)))
        self.FeatureNetwork = nn.Sequential(*CNNs)
        FCs = [nn.Linear(512, 1024),
               nn.ReLU(),
               nn.Linear(1024, 1024),
               nn.ReLU(),
               nn.Linear(1024, 2)
               ]
        self.fc = nn.Sequential(*FCs)

    def forward(self, xa, xb):
        fa = self.FeatureNetwork(xa)
        fb = self.FeatureNetwork(xb)
        f = torch.cat([fa, fb], dim=1)
        f = torch.flatten(f, start_dim=1)
        out = self.fc(f)

        return out


class Base_trans(nn.Module):
    # 输入必须是 64x64
    def __init__(self):
        self.name = "base_trans"
        super(Base_trans, self).__init__()
        A_bot = [CNN_Layer(3, 32, 3, padding=1),
                 CNN_Layer(32, 3, 3, padding=1)]
        B_bot = A_bot.copy()
        self.CNNA = nn.Sequential(*A_bot)
        self.CNNB = nn.Sequential(*B_bot)
        CNNs = []
        CNNs.append(CNN_Layer(6, 24, 7, padding=3, pooling=('max', 2, 2, 0)))
        CNNs.append(CNN_Layer(24, 64, 5, padding=2, pooling=('max', 3, 2, 0)))
        CNNs.append(CNN_Layer(64, 96, 3, padding=0))
        CNNs.append(CNN_Layer(96, 96, 1, padding=0))
        CNNs.append(CNN_Layer(96, 64, 3, strides=2,
                              padding=0, pooling=('max', 3, 2, 0)))
        self.FeatureNetwork = nn.Sequential(*CNNs)
        FCs = [nn.Linear(256, 1024),
               nn.ReLU(),
               nn.Linear(1024, 2),
               nn.PReLU()
               ]
        self.fc = nn.Sequential(*FCs)

    def forward(self, xa, xb):
        a = self.CNNA(xa)
        b = self.CNNB(xb)
        x = torch.cat([xa, xb], dim=1)
        f = self.FeatureNetwork(x)
        f = torch.flatten(f, start_dim=1)
        out = self.fc(f)
        return out


class TransMc(MatchNet):
    """
    OutPut: stacked features: shape [2, batchsize, feat_dim],
             A_feats and B_feats are in it
    """

    def __init__(self):
        super(TransMc, self).__init__()
        self.name = "TransMc"
        A_bot = [CNN_Layer(3, 32, 3, padding=1),
                 CNN_Layer(32, 3, 3, padding=1)]
        B_bot = A_bot.copy()
        self.CNNA = nn.Sequential(*A_bot)
        self.CNNB = nn.Sequential(*B_bot)
        CNNs = []
        CNNs.append(CNN_Layer(10, 24, 7, padding=3, pooling=('max', 2, 2, 0)))
        CNNs.append(CNN_Layer(24, 64, 5, padding=2, pooling=('max', 3, 2, 0)))
        CNNs.append(CNN_Layer(64, 96, 3, padding=0))
        CNNs.append(CNN_Layer(96, 96, 1, padding=0))
        CNNs.append(CNN_Layer(96, 64, 3, strides=2,
                              padding=0, pooling=('max', 3, 2, 0)))
        self.FeatureNetwork = nn.Sequential(*CNNs)
        FCs = [nn.Linear(256, 1024),
               nn.ReLU(),
               nn.Linear(1024, 2),
               nn.PReLU()
               ]
        self.fc = nn.Sequential(*FCs)

    def forward(self, xa, xb):
        a = self.CNNA(xa)
        b = self.CNNB(xb)
        # a = x[:,0]
        # b = x[:,1]
        c = self.rgb2gray(xa)
        d = self.rgb2gray(xb)
        e = self.sobel(c)
        f = self.sobel(d)
        x = torch.cat([a, c, e, b, d, f], dim=1)
        f = self.FeatureNetwork(x)
        f = torch.flatten(f, start_dim=1)
        out = self.fc(f)
        return out

    def rgb2gray(self, tensor):
        """ convert 3 channel rgb tensor to gray scale
            Input size : [bs, 3, h, w]
        """
        rgbWeight = torch.as_tensor([0.299,0.587,0.114],dtype=tensor.dtype,
                                    device=tensor.device).view(1,3,1,1)
        tensor = torch.sum(tensor*rgbWeight,dim=1,keepdim=True)

        # im = tensor[0].squeeze()
        # plt.figure()
        # plt.imshow(im, cmap=plt.cm.gray)
        # plt.show()

        return tensor

    def sobel(self, im):
        """ transfer image into gray scale
            im: Input size [bs, 1, h, w]
        """
        sobel_kernel = np.array(
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')  #
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        weight = torch.from_numpy(sobel_kernel).clone().detach().to(1)
        edge_detect = F.conv2d(im.clone().detach(), weight, padding=1)
        im = edge_detect[0].squeeze()

        return edge_detect


class BACN(MatchNet):
    """
    OutPut: stacked features: shape [2, batchsize, feat_dim],
             A_feats and B_feats are in it
    """

    def __init__(self):
        super(BACN, self).__init__()
        self.name = "DB_MC_CA"
        A_bot = [CNN_Layer(3, 32, 3, padding=1),
                 CNN_Layer(32, 3, 3, padding=1)]
        B_bot = A_bot.copy()
        self.CNNA = nn.Sequential(*A_bot)
        self.CNNB = nn.Sequential(*B_bot)
        CNNs = []
        CNNs.append(CNN_Layer(10, 24, 7, padding=3, pooling=('max', 2, 2, 0)))
        CNNs.append(CNN_Layer(24, 64, 5, padding=2, pooling=('max', 3, 2, 0)))
        CNNs.append(CNN_Layer(64, 96, 3, padding=0))
        CNNs.append(CNN_Layer(96, 96, 1, padding=0))
        CNNs.append(CNN_Layer(96, 64, 3, strides=2,
                              padding=0, pooling=('max', 3, 2, 0)))
        # self.FeatureNetwork = nn.Sequential(*CNNs)
        self.FeatureNetwork = nn.ModuleList(CNNs)
        FCs = [nn.Linear(256, 1024),
               nn.ReLU(),
               nn.Linear(1024, 2),
               nn.PReLU()
               ]
        self.fc = nn.Sequential(*FCs)
        self.ca = ChannelAttention(96)

    def forward(self, xa, xb):
        #with torch.autograd.set_detect_anomaly(True):
        a = self.CNNA(xa)
        b = self.CNNB(xb)
        # a = x[:,0]
        # b = x[:,1]
        c = self.rgb2gray(xa)
        d = self.rgb2gray(xb)
        e = self.sobel(c)
        f = self.sobel(d)
        x = torch.cat([a.to('cpu'), c.to('cpu'), e.to('cpu'), b.to('cpu'), d.to('cpu'), f.to('cpu')], dim=1)
        for m in self.FeatureNetwork[:4]:
            x = m(x)
        x = self.ca(x)
        for m in self.FeatureNetwork[4:]:
            x = m(x)
        x = torch.flatten(x, start_dim=1)
        out = self.fc(x)
        return out

    def rgb2gray(self, tensor):
        """ convert 3 channel rgb tensor to gray scale
            Input size : [bs, 3, h, w]
        """
        rgbWeight = torch.as_tensor([0.299,0.587,0.114],dtype=tensor.dtype,
                                    device=tensor.device).view(1,3,1,1)
        tensor = torch.sum(tensor*rgbWeight,dim=1,keepdim=True)

        # im = tensor[0].squeeze()
        # plt.figure()
        # plt.imshow(im, cmap=plt.cm.gray)
        # plt.show()

        return tensor

    def sobel(self, im):
        """ transfer image into gray scale
            im: Input size [bs, 1, h, w]
        """
        sobel_kernel = np.array(
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')  #
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        weight = torch.from_numpy(sobel_kernel).clone().detach().to('cpu')
        edge_detect = F.conv2d(im.clone().detach().to('cpu'), weight, padding=1)
        im = edge_detect[0].squeeze()

        # im = tensor[0].squeeze()
        # plt.figure()
        # plt.imshow(im, cmap=plt.cm.gray)
        # plt.show()
        # edge_detect = edge_detect.squeeze().detach().numpy()
        return edge_detect


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), 
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        w = self.sigmoid(avgout + maxout)
        return x * w.expand_as(x)



class Hog_FC(nn.Module):
    def __init__(self):
        self.name = 'hog_fc'
        super(Hog_FC, self).__init__()
        fcs=[nn.Linear(24576, 1024),#5832
            nn.ReLU(),
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Linear(256,2),
            nn.Dropout(0.4),
            ]
        self.fc = nn.Sequential(*fcs)
    
    def forward(self,x):
        return self.fc(x)

if __name__ == '__main__':
    a = torch.zeros(3, 3, 64, 64)
    b = torch.ones(3, 3, 64, 64)
    print(BACN()(a, b))
