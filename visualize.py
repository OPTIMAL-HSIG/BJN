import torch, cv2
from PIL import Image
import pandas as pd
import numpy as np
import torchvision.models as models
import matplotlib.pyplot as plt
from os.path import join as opj

from networks import *

bacnet = BACN()
bacnet.load_state_dict(torch.load('weights/{}_weight.pth'.format(bacnet.name)))

parm={}
for name,parameters in bacnet.named_parameters():
    print(name,':',parameters.size())
    parm[name]=parameters.detach().numpy()


def draw_handcraft():
    inps = 'pictures/samples/satellite/0000000410.jpg'
    img_DR = cv2.imread(inps,0)
    edges_DR = cv2.Canny(img_DR,20,40)
    sobel_DR = cv2.Sobel(img_DR,-1,0,1)
    bitnot_DR = cv2.bitwise_not(sobel_DR)
    # plt.imshow(Image.open(inps), cmap='gray')
    plt.imshow(img_DR, cmap = 'gray')
    plt.axis('off')
    plt.savefig('pictures/004.jpg')


if __name__ == "__main__":
   draw_handcraft() 