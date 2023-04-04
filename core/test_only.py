#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>

import logging
import torch
import torch.backends.cudnn
import torch.utils.data
from tqdm import tqdm
from time import time

import utils.data_loaders
import utils.data_transforms
import utils.helpers

import core.pipeline_test as pipeline

from models.encoder import Encoder
from models.decoder_pre_merger import Decoder
from models.refiner_3dresunet import Refiner
from models.merger_pre_merger import Merger


def test_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # load data
    taxonomies, test_data_loader = pipeline.load_data(cfg)

    # Set up networks
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    refiner = Refiner(cfg)
    merger = Merger(cfg)
    encoder, decoder, refiner, merger, _ = \
        pipeline.setup_network(cfg, encoder, decoder, refiner, merger)
    
    if cfg.CONST.IMB_WEIGHTS is not None:
        mlp = pipeline.setup_mlp(cfg)

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()

    for_tqdm = tqdm(enumerate(test_data_loader), total=n_samples)
    start_time = time()
    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume) in for_tqdm:
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]
        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images)
            ground_truth_volume = utils.helpers.var_or_cuda(ground_truth_volume)

            # Test the encoder, decoder, refiner and merger
            image_features = encoder.module.resnet_forward(rendering_images).unsqueeze(dim=0)

            if cfg.CONST.VIEWS_AFTER_REDUCTION is not None:
                if cfg.CONST.IMB_WEIGHTS is not None:
                    mapping_features = mlp(image_features.view(image_features.shape[1], 256, 56, 56))
                views = pipeline.reduce_branch_fps(mapping_features, cfg.CONST.VIEWS_AFTER_REDUCTION)
                image_features = image_features[:, views, ::]

            image_features = encoder.module.other_layers_forward(image_features.squeeze(dim=0), image_features.shape[1])
            raw_features, generated_volume = decoder(image_features)

            if cfg.NETWORK.USE_MERGER:
                generated_volume = merger(raw_features, generated_volume)
            else:
                generated_volume = torch.mean(generated_volume, dim=1)

            if cfg.NETWORK.USE_REFINER:
                generated_volume = refiner(generated_volume)

            # IoU per sample
            sample_iou = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                sample_iou.append((intersection / union).item())

            # IoU per taxonomy
            if taxonomy_id not in test_iou:
                test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
            test_iou[taxonomy_id]['n_samples'] += 1
            test_iou[taxonomy_id]['iou'].append(sample_iou)

            # Print sample loss and IoU
            if (sample_idx + 1) % 50 == 0:
                for_tqdm.update(50)
                for_tqdm.set_description('Test[%d/%d] Taxonomy = %s' %
                                         (sample_idx + 1, n_samples, taxonomy_id))

            logging.debug('Test[%d/%d] Taxonomy = %s Sample = %s IoU = %s' %
                          (sample_idx + 1, n_samples, taxonomy_id, sample_name, ['%.4f' % si for si in sample_iou]))

    print((time() - start_time))
    
    # Output testing results
    pipeline.output(cfg, test_iou, n_samples, taxonomies)
