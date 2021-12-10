import torch
import tqdm
import numpy as np
from networks import BACN, Baseline, TransMc, Base_trans, MatchNet
from dataset import Search_collection850
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def search_test(model=MatchNet, dev='cpu'):
    net = model().to(dev)
    # net.name = 'BACN'
    net.load_state_dict(torch.load(
        'weights/{}_weight.pth'.format(net.name),
        map_location=dev))
    net.eval()
    acc_1, acc_3, acc_5, total = 0,0,0,0
    for dd in tqdm.tqdm(Search_collection850(noisy=False)):
        total += 1
        confs = net(dd[0].to(dev), dd[1].to(dev))
        confs = torch.softmax(confs,1)[:,1].cpu().detach().numpy()
        args = [int(x) for x in (-confs).argsort()[:5]]
        labels = np.array(dd[3])
        if dd[2] == labels[args[0]]:
            acc_1 += 1
        if dd[2] in labels[args[:3]]:
            acc_3 += 1
        if dd[2] in labels[args[:5]]:
            acc_5 += 1

    print('{} -- top_1:{}, top3:{}, top5:{}'.format(
            net.name, acc_1/total, acc_3/total, acc_5/total))
        
        

if __name__ == '__main__':
    search_test()