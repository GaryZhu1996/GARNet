#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>

import os
import logging
import random
import math
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.helpers

from time import time

import core.pipeline_train as pipeline
from core.test import test_net

from models.encoder import Encoder

from models.decoder import Decoder
# from models.decoder_pre_merger_ebam import Decoder
# from models.decoder_pre_merger_ebam_relu import Decoder
# from models.decoder_pre_merger_ebam_64 import Decoder
# from models.decoder_pre_merger_ebam_128 import Decoder
# from models.decoder_pre_merger_bam import Decoder
# from models.decoder_pre_merger_bam_relu import Decoder
# from models.decoder_pre_merger_sa import Decoder
# from models.decoder_multi_merger_5fusion import Decoder
# from models.decoder_multi_merger_5fusion_cat import Decoder

from models.refiner import Refiner
# from models.refiner_resunet import Refiner
# from models.refiner_resunet_64 import Refiner
# from models.refiner_resunet_128 import Refiner
# from models.refiner_resunet_128_128channel import Refiner
# from models.refiner_resunet_bi import Refiner

from models.merger import Merger
# from models.merger_concat_res_pre_merger import Merger
# from models.merger_concat_res_pre_merger_64 import Merger
# from models.merger_concat_res_pre_merger_128 import Merger
# from models.merger_concat_pre_merger import Merger
# from models.merger_concat_res_pre_merger_with_supervise import Merger
# from models.merger_concat_res_multi_level_fusion import Merger
# from models.merger_multi_level_se import Merger

from losses.soft_dice_loss import SoftDiceLoss
from losses.precision_loss import PrecisionLoss
from losses.pr_loss import PRLoss
from utils.average_meter import AverageMeter


def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # load data
    train_data_loader, val_data_loader = pipeline.load_data(cfg)

    # load model
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    refiner = Refiner(cfg)
    merger = Merger(cfg)
    init_epoch, best_iou, best_epoch, encoder, decoder, refiner, merger, cfg = \
        pipeline.setup_network(cfg, encoder, decoder, refiner, merger)

    # Set up solver
    if cfg.NETWORK.MERGER_TYPE == 1:
        encoder_solver, decoder_solver, refiner_solver, merger_solver = \
            pipeline.solver(cfg, encoder, decoder, refiner, merger)
    else:
        encoder_solver, decoder_solver, pre_merger_solver, refiner_solver, merger_solver = \
            pipeline.solver(cfg, encoder, decoder, refiner, merger)
    
    # set up learning rate scheduler with warm up
    lambda_encoder = lambda cur_iter: (cur_iter + 1) / cfg.TRAIN.WARM_UP_EPOCH \
        if cur_iter < cfg.TRAIN.WARM_UP_EPOCH \
        else cfg.TRAIN.GAMMA ** len([m for m in cfg.TRAIN.ENCODER_LR_MILESTONES if m <= cur_iter])
    lambda_decoder = lambda cur_iter: (cur_iter + 1) / cfg.TRAIN.WARM_UP_EPOCH \
        if cur_iter < cfg.TRAIN.WARM_UP_EPOCH \
        else cfg.TRAIN.GAMMA ** len([m for m in cfg.TRAIN.DECODER_LR_MILESTONES if m <= cur_iter])
    lambda_refiner = lambda cur_iter: (cur_iter + 1) / cfg.TRAIN.WARM_UP_EPOCH \
        if cur_iter < cfg.TRAIN.WARM_UP_EPOCH \
        else cfg.TRAIN.GAMMA ** len([m for m in cfg.TRAIN.REFINER_LR_MILESTONES if m <= cur_iter])
    lambda_merger = lambda cur_iter: (cur_iter + 1) / cfg.TRAIN.WARM_UP_EPOCH \
        if cur_iter < cfg.TRAIN.WARM_UP_EPOCH \
        else cfg.TRAIN.GAMMA ** len([m for m in cfg.TRAIN.MERGER_LR_MILESTONES if m <= cur_iter])
    encoder_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(encoder_solver, lr_lambda=lambda_encoder)
    decoder_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(decoder_solver, lr_lambda=lambda_decoder)
    refiner_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(refiner_solver, lr_lambda=lambda_refiner)
    merger_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(merger_solver, lr_lambda=lambda_merger)
    if cfg.NETWORK.MERGER_TYPE != 1:
        lambda_pre_merger = lambda cur_iter: (cur_iter + 1) / cfg.TRAIN.WARM_UP_EPOCH \
            if cur_iter < cfg.TRAIN.WARM_UP_EPOCH \
            else cfg.TRAIN.GAMMA ** len([m for m in cfg.TRAIN.MERGER_LR_MILESTONES if m <= cur_iter])
        pre_merger_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(pre_merger_solver, lr_lambda=lambda_pre_merger)
    
    # Set up loss functions
    bce_loss = torch.nn.BCELoss()
    if cfg.TRAIN.SOFT_DIVE_LOSS_WEIGHT is not None:
        additional_loss = SoftDiceLoss()
        additional_loss_weight = cfg.TRAIN.SOFT_DIVE_LOSS_WEIGHT
    if cfg.TRAIN.PRECISION_LOSS_WEIGHT is not None:
        additional_loss = PrecisionLoss()
        additional_loss_weight = cfg.TRAIN.PRECISION_LOSS_WEIGHT
    if cfg.TRAIN.PR_LOSS_WEIGHT is not None:
        additional_loss = PRLoss()
        additional_loss_weight = cfg.TRAIN.PR_LOSS_WEIGHT

    # Summary writer for TensorBoard
    train_writer, val_writer = pipeline.setup_writer(cfg)
    
    # Training loop
    n_views_rendering = cfg.CONST.N_VIEWS_RENDERING
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHS):  # 0，250
        # Tick / tock
        epoch_start_time = time()
        
        # Batch average meterics
        batch_time = AverageMeter()
        data_time = AverageMeter()
        encoder_losses = AverageMeter()  # EDLoss
        refiner_losses = AverageMeter()  # RLoss

        # switch models to training mode
        encoder.train()
        decoder.train()
        merger.train()
        refiner.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)

        for batch_idx, (taxonomy_names, sample_names, rendering_images,
                        ground_truth_volumes) in enumerate(train_data_loader):
            # Measure data time
            if cfg.TRAIN.UPDATE_N_VIEWS_RENDERING_PER_ITERATION:
                n_views_rendering = random.randint(1, cfg.CONST.N_VIEWS_RENDERING)
            data_time.update(time() - batch_end_time)
            
            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images[:, :n_views_rendering, ::])
            ground_truth_volumes = utils.helpers.var_or_cuda(ground_truth_volumes)
            
            # Train the encoder, decoder, refiner, and merger
            # encoder
            image_features = encoder(rendering_images)
            
            # decoder
            if not (cfg.TRAIN.FIX_MERGER_FOR_1_VIEW and n_views_rendering == 1):
                if not cfg.NETWORK.BI_MERGER:
                    raw_features, generated_volumes = decoder(image_features)
                else:
                    raw_features, generated_volumes, fusion_features = decoder(image_features)
            else:
                if not cfg.NETWORK.BI_MERGER:
                    raw_features, generated_volumes = decoder(image_features, True)
                else:
                    raw_features, generated_volumes, fusion_features = decoder(image_features, True)

            # merger
            if not (cfg.TRAIN.FIX_MERGER_FOR_1_VIEW and n_views_rendering == 1):
                if cfg.TRAIN.MULTI_LEVEL_MEAN_SUPERVISE and cfg.TRAIN.MULTI_LEVEL_WEIGHTED_SUPERVISE:
                    generated_volumes, fusion_volumes = \
                        merger(raw_features, generated_volumes)
                    fusion_volumes = (generated_volumes + fusion_volumes) / 2
                elif cfg.TRAIN.MULTI_LEVEL_MEAN_SUPERVISE:
                    generated_volumes, fusion_volumes = \
                        merger(raw_features, generated_volumes)
                elif cfg.TRAIN.MULTI_LEVEL_WEIGHTED_SUPERVISE:
                    generated_volumes, fusion_volumes = \
                        merger(raw_features, generated_volumes, True)
                elif cfg.NETWORK.BI_MERGER:
                    generated_volumes = merger(raw_features, generated_volumes, fusion_features)
                else:
                    if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
                        generated_volumes = merger(raw_features, generated_volumes)
                    else:
                        generated_volumes = torch.mean(generated_volumes, dim=1)  # 没有启用merger，以下同理 取均值
            else:
                generated_volumes = torch.squeeze(generated_volumes)

            # ED Loss
            if (cfg.TRAIN.MULTI_LEVEL_MEAN_SUPERVISE or cfg.TRAIN.MULTI_LEVEL_WEIGHTED_SUPERVISE)\
                    and not (cfg.TRAIN.FIX_MERGER_FOR_1_VIEW and n_views_rendering == 1):
                encoder_loss = bce_loss(generated_volumes if fusion_volumes is None else fusion_volumes,
                                        ground_truth_volumes) * 10
            elif cfg.NETWORK.BI_MERGER and not (cfg.TRAIN.FIX_MERGER_FOR_1_VIEW and n_views_rendering == 1):
                encoder_loss = bce_loss(generated_volumes,
                                        ground_truth_volumes.unsqueeze(dim=1).expand(-1, 2, -1, -1, -1)) * 10
            else:
                encoder_loss = bce_loss(generated_volumes, ground_truth_volumes) * 10
            
            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                # refiner
                if cfg.NETWORK.BI_MERGER and (cfg.TRAIN.FIX_MERGER_FOR_1_VIEW and n_views_rendering == 1):
                    generated_volumes = refiner(generated_volumes.unsqueeze(dim=1).expand(-1, 2, -1, -1, -1))
                else:
                    generated_volumes = refiner(generated_volumes)

                # R Loss
                if cfg.TRAIN.SOFT_DIVE_LOSS_WEIGHT is not None \
                        or cfg.TRAIN.PRECISION_LOSS_WEIGHT is not None \
                        or cfg.TRAIN.PR_LOSS_WEIGHT is not None:
                    add_loss = additional_loss(generated_volumes, ground_truth_volumes) * additional_loss_weight
                    b_loss = bce_loss(generated_volumes, ground_truth_volumes) * 10
                    refiner_loss = add_loss + b_loss
                    if batch_idx == 0 or (batch_idx + 1) % cfg.TRAIN.SHOW_TRAIN_STATE == 0:
                        logging.info('DiceLoss = %.4f BCELoss = %.4f' % (add_loss, b_loss))
                else:
                    refiner_loss = bce_loss(generated_volumes, ground_truth_volumes) * 10
            else:
                refiner_loss = encoder_loss
            
            # Gradient decent
            encoder.zero_grad()
            decoder.zero_grad()
            refiner.zero_grad()
            merger.zero_grad()
            
            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                encoder_loss.backward(retain_graph=True)
                refiner_loss.backward()
            else:
                encoder_loss.backward()
            
            encoder_solver.step()
            decoder_solver.step()
            refiner_solver.step()
            merger_solver.step()
            if cfg.NETWORK.MERGER_TYPE != 1:
                pre_merger_solver.step()
            
            # Append loss to average metrics
            encoder_losses.update(encoder_loss.item())
            refiner_losses.update(refiner_loss.item())
            
            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('EncoderDecoder/BatchLoss', encoder_loss.item(), n_itr)
            train_writer.add_scalar('Refiner/BatchLoss', refiner_loss.item(), n_itr)
            
            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            if batch_idx == 0 or (batch_idx + 1) % cfg.TRAIN.SHOW_TRAIN_STATE == 0:
                logging.info(
                    '[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) EDLoss = %.4f RLoss = %.4f'
                    % (epoch_idx + 1, cfg.TRAIN.NUM_EPOCHS, batch_idx + 1, n_batches, batch_time.val, data_time.val,
                       encoder_loss.item(), refiner_loss.item()))
                if cfg.NETWORK.MERGER_TYPE == 1:
                    print('LearningRate:\tencoder: %f | decoder: %f | refiner: %f | merger: %f' %
                          (encoder_lr_scheduler.optimizer.param_groups[0]['lr'],
                           decoder_lr_scheduler.optimizer.param_groups[0]['lr'],
                           refiner_lr_scheduler.optimizer.param_groups[0]['lr'],
                           merger_lr_scheduler.optimizer.param_groups[0]['lr']))
                else:
                    print('LearningRate:\tencoder: %f | decoder: %f | pre-merger: %f  | refiner: %f | merger: %f' %
                          (encoder_lr_scheduler.optimizer.param_groups[0]['lr'],
                           decoder_lr_scheduler.optimizer.param_groups[0]['lr'],
                           pre_merger_lr_scheduler.optimizer.param_groups[0]['lr'],
                           refiner_lr_scheduler.optimizer.param_groups[0]['lr'],
                           merger_lr_scheduler.optimizer.param_groups[0]['lr']))
            else:
                logging.debug(
                    '[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) EDLoss = %.4f RLoss = %.4f'
                    % (epoch_idx + 1, cfg.TRAIN.NUM_EPOCHS, batch_idx + 1, n_batches, batch_time.val, data_time.val,
                       encoder_loss.item(), refiner_loss.item()))
        
        # Adjust learning rate
        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()
        if cfg.NETWORK.MERGER_TYPE != 1:
            pre_merger_lr_scheduler.step()
        refiner_lr_scheduler.step()
        merger_lr_scheduler.step()
        
        # Append epoch loss to TensorBoard
        train_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx + 1)
        train_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx + 1)
        
        # Tick / tock
        epoch_end_time = time()
        logging.info('[Epoch %d/%d] EpochTime = %.3f (s) EDLoss = %.4f RLoss = %.4f' %
                     (epoch_idx + 1, cfg.TRAIN.NUM_EPOCHS, epoch_end_time - epoch_start_time,
                      encoder_losses.avg, refiner_losses.avg))

        # Update Rendering Views
        if cfg.TRAIN.UPDATE_N_VIEWS_RENDERING_PER_EPOCH:
            n_views_rendering = random.randint(1, cfg.CONST.N_VIEWS_RENDERING)
            train_data_loader.dataset.set_n_views_rendering(n_views_rendering)
            logging.info('Epoch [%d/%d] Update #RenderingViews to %d' %
                         (epoch_idx + 2, cfg.TRAIN.NUM_EPOCHS, n_views_rendering))
        
        # Validate the training models
        iou = test_net(cfg, epoch_idx + 1, val_data_loader, val_writer,
                       encoder, decoder, refiner, merger)
        
        # Save weights to file  #保存模型
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0 or iou > best_iou:
            file_name = 'checkpoint-epoch-%03d.pth' % (epoch_idx + 1)
            if iou > best_iou:
                best_iou = iou
                best_epoch = epoch_idx
                file_name = 'checkpoint-best.pth'
            
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            if not os.path.exists(cfg.DIR.CHECKPOINTS):
                os.makedirs(cfg.DIR.CHECKPOINTS)
            
            checkpoint = {
                'epoch_idx': epoch_idx,
                'best_iou': best_iou,
                'best_epoch': best_epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
            }
            if cfg.NETWORK.USE_REFINER:
                checkpoint['refiner_state_dict'] = refiner.state_dict()
            if cfg.NETWORK.USE_MERGER:
                checkpoint['merger_state_dict'] = merger.state_dict()
            
            torch.save(checkpoint, output_path)
            logging.info('Saved checkpoint to %s ...' % output_path)
    
    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()
