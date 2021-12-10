
# ========  Training ============
TRAIN_DATA_ROOT = r'D:/FromLiu/paper01/data/collection_10000'
# ground truth json file, a dict that key is UAV image name "xxx.bmp" 
# and corresponding value is satellite image name "xxx.jpg"
TRAIN_GT_PATH = r'D:/FromLiu/paper01/data/train_gt.json'


# ========   Verification  ===========
VER_DATA_ROOT = r'D:/FromLiu/paper01/data/collection_1'
VER_GT_PATH = r'D:/FromLiu/paper01/data/test_gt.json'
# VER_DATA_ROOT = r'D:/DataSet/3'
# VER_GT_PATH = r'D:/DataSet/test_gt.json'
# ===== Hyperparameters ==========
class Hyper():
    def __init__(self):
        self.batchsize = 64

hyper = Hyper()
