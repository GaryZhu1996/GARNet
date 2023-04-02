#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>

import numpy as np
import logging
import torch
import torch.backends.cudnn
import torch.utils.data
from tqdm import tqdm

import utils.data_loaders
import utils.data_transforms
import utils.helpers

import core.pipeline_test as pipeline

from models.encoder import Encoder
from models.decoder_pre_merger import Decoder
from models.refiner_3dresunet import Refiner
from models.merger_pre_merger import Merger

from losses.pr_loss import PRLoss
from utils.average_meter import AverageMeter


def test_net(cfg,
             epoch_idx=-1,
             test_data_loader=None,
             test_writer=None,
             encoder=None,
             decoder=None,
             refiner=None,
             merger=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # load data
    taxonomies, test_data_loader = pipeline.load_data(cfg, test_data_loader)

    # Set up networks
    if decoder is None or encoder is None:
        encoder = Encoder(cfg)
        decoder = Decoder(cfg)
        refiner = Refiner(cfg)
        merger = Merger(cfg)

        encoder, decoder, refiner, merger, epoch_idx = \
            pipeline.setup_network(cfg, encoder, decoder, refiner, merger)

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()
    if cfg.TRAIN.PR_LOSS_WEIGHT is not None:
        additional_loss = PRLoss()
        additional_loss_weight = cfg.TRAIN.PR_LOSS_WEIGHT

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()
    encoder_losses = AverageMeter()
    refiner_losses = AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()

    for_tqdm = tqdm(enumerate(test_data_loader), total=n_samples)
    # for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume) in enumerate(test_data_loader):
    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume) in for_tqdm:
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]
        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images)
            ground_truth_volume = utils.helpers.var_or_cuda(ground_truth_volume)

            # Test the encoder, decoder, refiner and merger
            image_features = encoder(rendering_images)
            raw_features, generated_volume = decoder(image_features)

            if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
                generated_volume = merger(raw_features, generated_volume)
            else:
                generated_volume = torch.mean(generated_volume, dim=1)

            encoder_loss = bce_loss(generated_volume, ground_truth_volume) * 10

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volume = refiner(generated_volume)
                refiner_loss = bce_loss(generated_volume, ground_truth_volume) * 10
            else:
                refiner_loss = encoder_loss

            # Append loss and accuracy to average metrics
            encoder_losses.update(encoder_loss.item())
            refiner_losses.update(refiner_loss.item())

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

            # Append generated volumes to TensorBoard
            if test_writer and sample_idx < 3 and epoch_idx % 10:
                # Volume Visualization
                rendering_views = utils.helpers.get_volume_views(generated_volume.cpu().numpy())
                test_writer.add_image('Model%02d/Reconstructed' % sample_idx, rendering_views, epoch_idx)
                rendering_views = utils.helpers.get_volume_views(ground_truth_volume.cpu().numpy())
                test_writer.add_image('Model%02d/GroundTruth' % sample_idx, rendering_views, epoch_idx)

            # Print sample loss and IoU
            if (sample_idx + 1) % 50 == 0:
                for_tqdm.update(50)
                for_tqdm.set_description('Test[%d/%d] Taxonomy = %s EDLoss = %.4f RLoss = %.4f' %
                                         (sample_idx + 1, n_samples, taxonomy_id,
                                          encoder_losses.avg, refiner_losses.avg))

            logging.debug('Test[%d/%d] Taxonomy = %s Sample = %s EDLoss = %.4f RLoss = %.4f IoU = %s' %
                          (sample_idx + 1, n_samples, taxonomy_id, sample_name,
                           encoder_loss.item(), refiner_loss.item(), ['%.4f' % si for si in sample_iou]))

    # Output testing results
    mean_iou = pipeline.output(cfg, test_iou, n_samples, taxonomies)

    # Add testing results to TensorBoard
    max_iou = np.max(mean_iou)
    if test_writer is not None:
        test_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx)
        test_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx)
        test_writer.add_scalar('Refiner/IoU', max_iou, epoch_idx)

    return max_iou
