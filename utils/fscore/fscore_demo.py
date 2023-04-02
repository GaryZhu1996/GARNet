import json
import os
from collections import OrderedDict

import numpy as np
import logging
import torch
import torch.backends.cudnn
import torch.utils.data
import open3d
import cv2
import utils.data_loaders
import utils.data_transforms
import utils.helpers
import utils.binvox_visualization
import utils.binvox_rw
from time import time

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger
from utils.average_meter import AverageMeter
from config import cfg
from PIL import Image
from utils.voxel import voxel2obj
from utils.fscore.util import voxel_grid_to_mesh, calculate_fscore, visualize_distance


def test_net():
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    refiner = Refiner(cfg)
    merger = Merger(cfg)
    refiner = torch.nn.DataParallel(refiner)
    cfg.CONST.WEIGHTS = '/disk1/zzw/works/pix2vox++/output/checkpoints/2021-06-04T11:00:55.871428/checkpoint-best.pth'
    print('Loading weights from %s ...' % cfg.CONST.WEIGHTS)
    checkpoint = torch.load(cfg.CONST.WEIGHTS, map_location=torch.device('cpu'))
    
    fix_checkpoint = {'encoder_state_dict': OrderedDict((k.split('module.')[1:][0], v)
                                                        for k, v in checkpoint['encoder_state_dict'].items()),
                      'decoder_state_dict': OrderedDict((k.split('module.')[1:][0], v)
                                                        for k, v in checkpoint['decoder_state_dict'].items()),
                      'refiner_state_dict': OrderedDict((k.split('module.')[1:][0], v)
                                                        for k, v in checkpoint['refiner_state_dict'].items()),
                      'merger_state_dict': OrderedDict((k.split('module.')[1:][0], v)
                                                       for k, v in checkpoint['merger_state_dict'].items())}

    epoch_idx = checkpoint['epoch_idx']
    encoder.load_state_dict(fix_checkpoint['encoder_state_dict'])
    decoder.load_state_dict(fix_checkpoint['decoder_state_dict'])
    
    if cfg.NETWORK.USE_REFINER:
        print('Use refiner')
        refiner.load_state_dict(checkpoint['refiner_state_dict'])
    if cfg.NETWORK.USE_MERGER:
        print('Use merger')
        merger.load_state_dict(fix_checkpoint['merger_state_dict'])
    
    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()
    
    # img1_path = './imgs/a.png'
    # img1_np = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
    volume_path = './test/part2/airplane/multi/voxel/model.binvox'
    img1_path = './test/part2/airplane/multi/'
    img1_np = []
    # img1_np = cv2.imread(img1_path)
    for i in range(3):
        img1 = cv2.imread(img1_path + '%d.png' % i, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
        # img1_np.append([np.array(img1).transpose((2, 0, 1)).astype(np.float32) / 255])
        img1_np.append(img1)
    
    sample = np.asarray(img1_np)
    # sample = np.array(img1_np)
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    
    test_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])
    
    rendering_images = test_transforms(rendering_images=sample)
    rendering_images = rendering_images.unsqueeze(0)
    
    with torch.no_grad():
        image_features = encoder(rendering_images)
        raw_features, generated_volume = decoder(image_features)
        
        if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
            generated_volume = merger(raw_features, generated_volume)
        else:
            generated_volume = torch.mean(generated_volume, dim=1)
        
        if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
            generated_volume = refiner(generated_volume)
    
    generated_volume_f = generated_volume.squeeze(0)
    generated_volume_num = generated_volume_f.cpu().numpy()
    generated_volume_num = (generated_volume_num > 0.4)
    
    with open(volume_path, 'rb') as f:
        ground_truth_volume = utils.binvox_rw.read_as_3d_array(f)
        ground_truth_volume = torch.from_numpy(ground_truth_volume.data.astype(np.float32)).cuda()
    mesh_time = time()
    generated_f = voxel_grid_to_mesh(generated_volume_num)
    ground_f = voxel_grid_to_mesh(ground_truth_volume.cpu().numpy())
    meshtime = time() - mesh_time
    # visualization mesh
    # open3d.visualization.draw_geometries([generated_f])
    # open3d.visualization.draw_geometries([ground_f])
    sample_time = time()
    generated_f_sample = open3d.geometry.TriangleMesh.sample_points_poisson_disk(generated_f, 8192)
    ground_f_sample = open3d.geometry.TriangleMesh.sample_points_poisson_disk(ground_f, 8192)
    sampletime = time() - sample_time
    fscore_time = time()
    fs, pr, re = calculate_fscore(ground_f_sample, generated_f_sample, th=0.01)
    fscoretime = time() - fscore_time
    print(
        'Fscore: %.4f, mesh time: %.3f, sample time: %.3f, fscore time: %.3f' % (fs, meshtime, sampletime, fscoretime))
    # visualization fscore
    # open3d.visualization.draw_geometries([ground_f_sample])
    # open3d.visualization.draw_geometries([generated_f_sample])
    # visualize_distance(ground_f_sample, generated_f_sample, max_distance=0.01)
    
    _volume = torch.ge(generated_volume, 0.4).float()
    intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
    union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
    iou = (intersection / union).item()
    print('%.4f' % iou)
    
    # Mintersection = torch.sum(_volume.mul(ground_truth_volume)).float()
    # Munion = torch.sum(ground_truth_volume).float()
    # Miou = (Mintersection / Munion).item()
    # print('%.4f' % Miou)
    
    img_dir = './imgs/result'
    gv = generated_volume_f.cpu().numpy()
    gv_new = np.swapaxes(gv, 2, 1)
    rendering_views = utils.binvox_visualization.get_volume_views(gv_new, os.path.join(img_dir),
                                                                  epoch_idx)
    voxel2obj('prediction.obj', gv > 0.4)


if __name__ == '__main__':
    # Set the batch size to 1
    test_net()
