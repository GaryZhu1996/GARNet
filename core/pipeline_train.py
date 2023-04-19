#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>

import numpy as np
import os
import torch
import logging
import utils.data_loaders
import utils.data_transforms
import utils.helpers
from tensorboardX import SummaryWriter
from datetime import datetime as dt


def modify_lr_strategy(cfg, current_epoch):
    milestone_lists = [cfg.TRAIN.ENCODER_LR_MILESTONES, cfg.TRAIN.DECODER_LR_MILESTONES]
    init_lr_list = [cfg.TRAIN.ENCODER_LEARNING_RATE, cfg.TRAIN.DECODER_LEARNING_RATE]
    if cfg.NETWORK.USE_REFINER:
        milestone_lists.append(cfg.TRAIN.REFINER_LR_MILESTONES)
        init_lr_list.append(cfg.TRAIN.REFINER_LEARNING_RATE)
    if cfg.NETWORK.USE_MERGER:
        milestone_lists.append(cfg.TRAIN.MERGER_LR_MILESTONES)
        init_lr_list.append(cfg.TRAIN.MERGER_LEARNING_RATE)
    current_milestone_list = []
    current_epoch_lr_list = []
    for milestones, init_lr in zip(milestone_lists, init_lr_list):
        milestones = np.array(milestones) - current_epoch
        init_lr = init_lr * cfg.TRAIN.GAMMA ** len(np.where(milestones <= 0)[0])
        milestones = list(milestones[len(np.where(milestones <= 0)[0]):])
        current_milestone_list.append(milestones)
        current_epoch_lr_list.append(init_lr)
    cfg.TRAIN.ENCODER_LR_MILESTONES = current_milestone_list[0]
    cfg.TRAIN.DECODER_LR_MILESTONES = current_milestone_list[1]
    cfg.TRAIN.ENCODER_LEARNING_RATE = current_epoch_lr_list[0]
    cfg.TRAIN.DECODER_LEARNING_RATE = current_epoch_lr_list[1]
    if cfg.NETWORK.USE_REFINER:
        cfg.TRAIN.REFINER_LR_MILESTONES = current_milestone_list[2]
        cfg.TRAIN.REFINER_LEARNING_RATE = current_epoch_lr_list[2]
    if cfg.NETWORK.USE_MERGER:
        cfg.TRAIN.MERGER_LR_MILESTONES = current_milestone_list[3 if cfg.NETWORK.USE_REFINER else 2]
        cfg.TRAIN.MERGER_LEARNING_RATE = current_epoch_lr_list[3 if cfg.NETWORK.USE_REFINER else 2]
    return cfg


def load_data(cfg):
    # Set up data augmentation
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
        utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.RandomFlip(),
        utils.data_transforms.RandomPermuteRGB(),
        utils.data_transforms.ToTensor(),
    ])
    val_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)  # 默认都用shapenet
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.TRAIN,
            cfg.CONST.N_VIEWS_RENDERING,
            train_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER,
        pin_memory=True,
        shuffle=True,
        drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.VAL,
            cfg.CONST.N_VIEWS_RENDERING,
            val_transforms),
        batch_size=1,
        num_workers=cfg.CONST.NUM_WORKER,
        pin_memory=True,
        shuffle=False)

    return train_data_loader, val_data_loader


def setup_network(cfg, encoder, decoder, refiner, merger):
    # Set up networks
    logging.info('Parameters in Encoder: %d.' % (utils.helpers.count_parameters(encoder)))
    logging.info('Parameters in Decoder: %d.' % (utils.helpers.count_parameters(decoder)))
    logging.info('Parameters in Refiner: %d.' % (utils.helpers.count_parameters(refiner)))
    logging.info('Parameters in Merger: %d.' % (utils.helpers.count_parameters(merger)))

    # Initialize weights of networks
    encoder.apply(utils.helpers.init_weights)
    decoder.apply(utils.helpers.init_weights)
    refiner.apply(utils.helpers.init_weights)
    merger.apply(utils.helpers.init_weights)
    if cfg.NETWORK.INIT_BACKBONE:
        encoder.init_backbone()

    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        refiner = torch.nn.DataParallel(refiner).cuda()
        merger = torch.nn.DataParallel(merger).cuda()

        # Load pretrained model if exists
        init_epoch = 0
        best_iou = -1
        best_epoch = -1
        if cfg.TRAIN.RESUME_TRAIN and 'WEIGHTS' in cfg.CONST:
            logging.info('Recovering from %s ...' % cfg.CONST.WEIGHTS)
            checkpoint = torch.load(cfg.CONST.WEIGHTS)
            init_epoch = checkpoint['epoch_idx'] + 1
            # best_iou = checkpoint['best_iou']
            best_epoch = checkpoint['best_epoch']

            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
            if cfg.NETWORK.USE_REFINER:
                refiner.load_state_dict(checkpoint['refiner_state_dict'])
            if cfg.NETWORK.USE_MERGER and cfg.TRAIN.LOAD_MERGER:
                merger.load_state_dict(checkpoint['merger_state_dict'])

            logging.info('Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' %
                         (init_epoch, best_iou, best_epoch))

            # resume the learning-rate strategy
            cfg = modify_lr_strategy(cfg, init_epoch)
        
    return init_epoch, best_iou, best_epoch, encoder, decoder, refiner, merger, cfg


def setup_writer(cfg):
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
    cfg.DIR.LOGS = output_dir % 'logs'
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))
    return train_writer, val_writer


def solver(cfg, encoder, decoder, refiner, merger):
    if cfg.TRAIN.POLICY == 'adam':
        encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()),
                                          lr=cfg.TRAIN.ENCODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        if cfg.NETWORK.MERGER_TYPE == 1:
            decoder_solver = base_solver(cfg, decoder)
        elif cfg.NETWORK.MERGER_TYPE == 2:
            decoder_solver, pre_merger_solver = pre_merge_solver(cfg, decoder)
        elif cfg.NETWORK.MERGER_TYPE == 3:
            decoder_solver, pre_merger_solver = multi_merge_solver(cfg, decoder)
        refiner_solver = torch.optim.Adam(refiner.parameters(),
                                          lr=cfg.TRAIN.REFINER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        merger_solver = torch.optim.Adam(merger.parameters(),
                                         lr=cfg.TRAIN.MERGER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))
    
    if cfg.NETWORK.MERGER_TYPE == 1:
        return encoder_solver, decoder_solver, refiner_solver, merger_solver
    else:
        return encoder_solver, decoder_solver, pre_merger_solver, refiner_solver, merger_solver


def base_solver(cfg, decoder):
    decoder_solver = torch.optim.Adam(decoder.parameters(),
                                      lr=cfg.TRAIN.DECODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
    return decoder_solver
    

def pre_merge_solver(cfg, decoder):
    fusion_paras = list(map(id, decoder.module.fusion_module.parameters()))
    based_paras = filter(lambda p: id(p) not in fusion_paras, decoder.parameters())
    decoder_solver = torch.optim.Adam(based_paras,
                                      lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                      betas=cfg.TRAIN.BETAS)
    pre_merger_solver = torch.optim.Adam(decoder.module.fusion_module.parameters(),
                                         # lr=cfg.TRAIN.PRE_MERGER_LEARNING_RATE,
                                         lr=cfg.TRAIN.MERGER_LEARNING_RATE,
                                         betas=cfg.TRAIN.BETAS)
    return decoder_solver, pre_merger_solver


def multi_merge_solver(cfg, decoder):
    fusion_parts = list(map(id, decoder.module.first_fusion_module.parameters())) + \
                   list(map(id, decoder.module.fusion_modules[0].parameters())) + \
                   list(map(id, decoder.module.fusion_modules[1].parameters())) + \
                   list(map(id, decoder.module.fusion_modules[2].parameters()))
    # fusion_parts = list(map(id, decoder.module.first_fusion_module.parameters())) + \
    #                list(map(id, decoder.module.second_fusion_module.parameters()))
    based_paras = filter(lambda p: id(p) not in fusion_parts, decoder.parameters())
    fusion_paras = filter(lambda p: id(p) in fusion_parts, decoder.parameters())
    decoder_solver = torch.optim.Adam(based_paras,
                                      lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                      betas=cfg.TRAIN.BETAS)
    pre_merger_solver = torch.optim.Adam(fusion_paras,
                                         lr=cfg.TRAIN.MERGER_LEARNING_RATE,
                                         betas=cfg.TRAIN.BETAS)
    
    return decoder_solver, pre_merger_solver
