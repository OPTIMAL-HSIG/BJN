'''
Author: your name
Date: 2020-11-21 20:29:41
LastEditTime: 2020-11-22 10:50:38
LastEditors: Liu Chen
Description: 
FilePath: \BACN\train.py
  
'''
import argparse
import torch, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset import Train_Collection
from networks import MatchNet, BACN, SplitNet
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from config import *
from test import Verification_test_BACN, Verification_test_MatchNet

parser = argparse.ArgumentParser(description='PyTorch BACN Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--num_epochs', default=100, type=int, help='num_epochs')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--gpu_id', default=0, type=int, help='gpu_id')
parser.add_argument('--dataParaller', '-d', action='store_true', help='dataParaller')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
args = parser.parse_args()
# if(args.dataParaller):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# else:
#     cudaName = "cuda:"+str(args.gpu_id)
#     device = torch.device(cudaName if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")

writer = SummaryWriter(log_dir='vislog')

Train_dataset = Train_Collection()
DS = DataLoader(Train_dataset, args.batchsize)


class Balance_loss(nn.Module):
    def __init__(self):
        super(Balance_loss, self).__init__()
        self.margin = 10
        self.eps = 1e-9

    def forward(self, predsA, predsB, target):
        distances = (predsA - predsB).pow(2).sum(1)
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() *
                        F.relu(self.margin - (distances + self.eps).sqrt()).pow(2)
                        )
        return losses.sum()

start_epoch = 0
num_epo = args.num_epochs
# net = SplitNet().to(device)
net = MatchNet().to(device)
# net = BACN().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
# optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
# scheduler = optimizer.lr_scheduler.MultiStepLR(optimizer, 
#                                         milestones=[int(0.4 * num_epo), int(0.7 * num_epo),
#                                         int(0.8 * num_epo),int(0.9 * num_epo)], gamma=0.1)
model_save_path = 'weights/{}_weight.pth'.format(net.name)
if args.resume:
    print('====> Resuming from checkpoint..\n')
    checkpoint = torch.load(model_save_path)
    net.load_state_dict(checkpoint)
# lossfunc = Balance_loss()
lossfunc = torch.nn.CrossEntropyLoss()

itt = 0
globa_ver = 0
best_iter = 0

for epoch in range(start_epoch, start_epoch+num_epo):
    for a, b, labels in DS:
        net.train()
        a = a.to(device)
        b = b.to(device)
        labels = labels.to(device)
        # ## ======== MatchNet =====
        x = net(a,b)
        xplus = net(a,b)
        loss = lossfunc(x, labels.squeeze().long())

        # ======== BACN  =====
        # out = net(a, b)
        # loss = lossfunc(out, labels.squeeze().long())

        loss.backward()
        optimizer.step()
        print(float(loss.data), end=' ')
        writer.add_scalar('scalar/loss', float(loss.data), itt)
        itt += 1
        if itt % 10 == 0:
            ver_rate = Verification_test_MatchNet(net, device=device)
            # ver_rate = Verification_test_BACN(net, device=device)
            if ver_rate >= globa_ver:
                globa_ver = ver_rate
                best_iter = itt
                torch.save(net.state_dict(), model_save_path)
            print('best: ', globa_ver, 'iter: ', best_iter)   

writer.close()
