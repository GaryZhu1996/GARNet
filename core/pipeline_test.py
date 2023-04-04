#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>

import torch
import numpy as np
import json
import logging

import utils.data_loaders
import utils.data_transforms
import utils.helpers


def reduce_branch_fps(features, left_view_num):
    samples = []
    remaining = list(range(features.shape[0]))
    # center of gravity7.176
    center = features.mean(dim=0).unsqueeze(dim=0)

    # sampling the first point
    view = euclidean_distances(center, features).squeeze().argmax()
    samples.append(remaining[view])
    remaining.pop(view)
    for _ in range(1, left_view_num):
        view = torch.min(euclidean_distances(features[samples, :], features[remaining, :]), dim=0)[0].argmax()
        samples.append(remaining[view])
        remaining.pop(view)
    samples.sort()
    return samples


def euclidean_distances(a, b):
    sq_a = a ** 2
    sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m->[m, 1]
    sq_b = b ** 2
    sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)  # n->[1, n]
    bt = b.t()
    return torch.sqrt(sum_sq_a + sum_sq_b - 2 * a.mm(bt))


def load_data(cfg, test_data_loader=None, random_for_reduce_view=None):
    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}
    
    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])
        
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=dataset_loader.get_dataset(
                utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING,
                test_transforms, random_for_reduce_view),
            batch_size=1,
            num_workers=cfg.CONST.NUM_WORKER,
            pin_memory=True,
            shuffle=False)
    return taxonomies, test_data_loader


def setup_network(cfg, encoder, decoder, refiner, merger):
    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        refiner = torch.nn.DataParallel(refiner).cuda()
        merger = torch.nn.DataParallel(merger).cuda()
    
    logging.info('Loading weights from %s ...' % cfg.CONST.WEIGHTS)
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    epoch_idx = checkpoint['epoch_idx']
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    if cfg.NETWORK.USE_REFINER:
        refiner.load_state_dict(checkpoint['refiner_state_dict'])
    if cfg.NETWORK.USE_MERGER:
        merger.load_state_dict(checkpoint['merger_state_dict'])
        
    return encoder, decoder, refiner, merger, epoch_idx


def setup_mlp(cfg):
    print(cfg.CONST.IMB_WEIGHTS)
    mlp = torch.load(cfg.CONST.IMB_WEIGHTS).eval()
    return mlp


def output(cfg, test_iou, n_samples, taxonomies):
    mean_iou = []
    for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
    mean_iou = np.sum(mean_iou, axis=0) / n_samples
    
    # Print header
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    print('Baseline', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    # Print body
    for taxonomy_id in test_iou:
        print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
        print('%d' % test_iou[taxonomy_id]['n_samples'], end='\t')
        # if 'baseline' in taxonomies[taxonomy_id]:
        #     print('%.4f' % taxonomies[taxonomy_id]['baseline']['%d-view' % cfg.CONST.N_VIEWS_RENDERING], end='\t\t')
        # else:
        print('N/a', end='\t\t')
        
        for ti in test_iou[taxonomy_id]['iou']:
            print('%.4f' % ti, end='\t')
        print()
    # Print mean IoU for each threshold
    print('Overall ', end='\t\t\t\t')
    for mi in mean_iou:
        print('%.4f' % mi, end='\t')
    print('\n')
    
    return mean_iou


def seed_output(cfg, SEED, mean_seed):
    mean_iou = np.sum(mean_seed, axis=0) / len(SEED)
    print('============================ SEED RESULTS ============================')
    print('Seed', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    for i in range(len(mean_seed)):
        print('%d' % SEED[i], end='\t')
        for j in range(len(mean_seed[i])):
            print('%.4f' % mean_seed[i][j], end='\t')
        print()
    print('Mean ', end='\t')
    for mi in mean_iou:
        print('%.4f' % mi, end='\t')
    print('\n')
