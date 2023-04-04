# -*- coding: utf-8 -*-
# 
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>

from easydict import EasyDict as edict


__C = edict()
cfg = __C


#
# Dataset Config
#
__C.DATASETS = edict()
__C.DATASETS.SHAPENET = edict()
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH = './datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.RENDERING_PATH = '/home/zzw/datasets/ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.VOXEL_PATH = '/home/zzw/datasets/ShapeNet/ShapeNetVox32/%s/%s/model.binvox'


#
# Dataset
#
__C.DATASET = edict()
__C.DATASET.MEAN = [0.5, 0.5, 0.5]
__C.DATASET.STD = [0.5, 0.5, 0.5]
__C.DATASET.TRAIN_DATASET = 'ShapeNet'
__C.DATASET.TEST_DATASET = 'ShapeNet'


#
# Common
#
__C.CONST = edict()
__C.CONST.DEVICE = '1'
__C.CONST.RNG_SEED = 0
__C.CONST.IMG_W = 224  # Image width for input
__C.CONST.IMG_H = 224  # Image height for input
__C.CONST.CROP_IMG_W = 128  # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_H = 128  # Dummy property for Pascal 3D
__C.CONST.BATCH_SIZE = 64  # for train only
__C.CONST.N_VIEWS_RENDERING = 3
__C.CONST.NUM_WORKER = 30  # number of data workers
__C.CONST.WEIGHTS = 'pths/garnet.pth'
__C.CONST.VIEWS_AFTER_REDUCTION = None  # for test only
__C.CONST.IMB_WEIGHTS = None  # 'pths/imb_for_garnet.pth'


#
# Directories
#
__C.DIR = edict()
__C.DIR.OUT_PATH = './output'


#
# Network
#
__C.NETWORK = edict()
__C.NETWORK.LEAKY_VALUE = .2
__C.NETWORK.TCONV_USE_BIAS = False
__C.NETWORK.INIT_BACKBONE = True  # use pretrained resnet50
__C.NETWORK.USE_REFINER = True
__C.NETWORK.USE_MERGER = True

__C.NETWORK.MERGER_TYPE = 2  # 1 for base; 2 for pre-merger  # for train only


#
# Training
#
__C.TRAIN = edict()
__C.TRAIN.RESUME_TRAIN = False
__C.TRAIN.LOAD_MERGER = False
__C.TRAIN.NUM_EPOCHS = 200
__C.TRAIN.BRIGHTNESS = .4
__C.TRAIN.CONTRAST = .4
__C.TRAIN.SATURATION = .4
__C.TRAIN.NOISE_STD = .1
__C.TRAIN.RANDOM_ROTATION = 30
__C.TRAIN.RANDOM_BG_COLOR_RANGE = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.POLICY = 'adam'  # available options: sgd, adam
__C.TRAIN.EPOCH_START_USE_REFINER = 0
__C.TRAIN.EPOCH_START_USE_MERGER = 0

__C.TRAIN.ENCODER_LEARNING_RATE = 1e-3
__C.TRAIN.DECODER_LEARNING_RATE = 1e-3
__C.TRAIN.REFINER_LEARNING_RATE = 1e-3
__C.TRAIN.MERGER_LEARNING_RATE = 3e-3/2

__C.TRAIN.ENCODER_LR_MILESTONES = [40, 60, 80, 100, 140, 180]
__C.TRAIN.DECODER_LR_MILESTONES = [40, 60, 80, 100, 140, 180]
__C.TRAIN.REFINER_LR_MILESTONES = [40, 60, 80, 100, 140, 180]
__C.TRAIN.MERGER_LR_MILESTONES = [40, 60, 80, 100, 140, 180]

__C.TRAIN.WARM_UP_EPOCH = 0
__C.TRAIN.BETAS = (.9, .999)
__C.TRAIN.MOMENTUM = .9
__C.TRAIN.GAMMA = .1
__C.TRAIN.SAVE_FREQ = 10  # weights will be overwritten every save_freq epoch
__C.TRAIN.SHOW_TRAIN_STATE = 1000
__C.TRAIN.UPDATE_N_VIEWS_RENDERING_PER_EPOCH = False
__C.TRAIN.UPDATE_N_VIEWS_RENDERING_PER_ITERATION = True
__C.TRAIN.FIX_MERGER_FOR_1_VIEW = True

__C.TRAIN.PR_LOSS_WEIGHT = None  # None or 0.5


#
# Testing options
#
__C.TEST = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH = [.2, .3, .4, .5]


#
# Reduce branches
#
__C.REDUCE = edict()
__C.REDUCE.MLP_DIM = [1024, 256, 5]
__C.REDUCE.MATRIX_OPERATION = False
__C.REDUCE.LEARNING_RATE = 2e-3
__C.REDUCE.GAMMA = .1
__C.REDUCE.MILESTONES = [5, 10]  # [5, 10, 15]
__C.REDUCE.NUM_EPOCHS = 15
__C.REDUCE.SAVE_FREQ = 1
__C.REDUCE.SHOW_TRAIN_STATE = 300
