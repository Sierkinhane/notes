# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn), Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = 'output'
_C.LOG_DIR = 'log'
_C.GPUS = (0,)
_C.WORKERS = 16
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.TARGET_TYPE = 'Gaussian'
_C.MODEL.IMAGE_SIZE = [256, 256]  # width * height
_C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height
_C.MODEL.SIGMA = 1.5
_C.MODEL.EXTRA = CN()

# High-Resoluion Net
_C.MODEL.EXTRA.PRETRAINED_LAYERS = ['*']
_C.MODEL.EXTRA.STEM_INPLANES = 64
_C.MODEL.EXTRA.FINAL_CONV_KERNEL = 1
_C.MODEL.EXTRA.WITH_HEAD = True

_C.MODEL.EXTRA.STAGE2 = CN()
_C.MODEL.EXTRA.STAGE2.NUM_MODULES = 1
_C.MODEL.EXTRA.STAGE2.NUM_BRANCHES = 2
_C.MODEL.EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
_C.MODEL.EXTRA.STAGE2.NUM_CHANNELS = [18, 36]
_C.MODEL.EXTRA.STAGE2.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE2.FUSE_METHOD = 'SUM'

_C.MODEL.EXTRA.STAGE3 = CN()
_C.MODEL.EXTRA.STAGE3.NUM_MODULES = 1
_C.MODEL.EXTRA.STAGE3.NUM_BRANCHES = 3
_C.MODEL.EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
_C.MODEL.EXTRA.STAGE3.NUM_CHANNELS = [18, 36, 72]
_C.MODEL.EXTRA.STAGE3.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE3.FUSE_METHOD = 'SUM'

_C.MODEL.EXTRA.STAGE4 = CN()
_C.MODEL.EXTRA.STAGE4.NUM_MODULES = 1
_C.MODEL.EXTRA.STAGE4.NUM_BRANCHES = 4
_C.MODEL.EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
_C.MODEL.EXTRA.STAGE4.NUM_CHANNELS = [18, 32, 72, 144]
_C.MODEL.EXTRA.STAGE4.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE4.FUSE_METHOD = 'SUM'

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = 'H:/DL-DATASET/68-LANDMARK-WFLW_OHKPM/WFLW_images/WFLW_images'
_C.DATASET.DATASET = 'WFLW_OHKPM'
_C.DATASET.TRAINSET = 'H:/DL-DATASET/HRNET-facial-landmark/data/data/wflw/face_landmarks_wflw_train.csv'
_C.DATASET.TESTSET = 'H:/DL-DATASET/HRNET-facial-landmark/data/data/wflw/face_landmarks_wflw_test_largepose.csv'

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [30, 50]
_C.TRAIN.LR = 0.0001

_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0.0
_C.TRAIN.WD = 0.0
_C.TRAIN.NESTEROV = False

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 60

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''#'/media/hane/DataBase/AI-PROJECTS/HRNet-Facial-Landmark-Detection-master/HRNet-Facial-Landmark-Detection-master/output/WFLW/face_alignment_wflw_hrnet_w18/checkpoint_2.pth'

_C.TRAIN.BATCH_SIZE_PER_GPU = 16
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
# Test Model Epoch
_C.TEST.FLIP_TEST = False


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args)
    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
