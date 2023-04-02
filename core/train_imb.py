#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>

import torch
import os
from time import time
from datetime import datetime as dt
import matplotlib.pyplot as plt

from config import cfg
import utils.helpers
from utils.average_meter import AverageMeter

import core.pipeline_train as pipe_train
import core.pipeline_test as pipe_test

from models.encoder import Encoder
from models.decoder_pre_merger import Decoder
from models.refiner_3dresunet import Refiner
from models.merger_pre_merger import Merger

from models.imb import MLP


inference_model_path = './pth/garnet.pth'
results_dir = './output/imb_weights/'


def dice_distance(a, b):
    intersection = a.view(a.shape[0], a.shape[1], -1).matmul(b.view(b.shape[0], b.shape[1], -1).permute(0, 2, 1))
    sum_sq_a = torch.sum(torch.square(a), dim=(2, 3, 4)).unsqueeze(2)  # m->[b, m, 1]
    sum_sq_b = torch.sum(torch.square(b), dim=(2, 3, 4)).unsqueeze(1)  # n->[b, 1, n]
    dice = 2. * intersection / (sum_sq_a + sum_sq_b)  # * 100
    return 1 - dice


def euclidean_distances(a, b):
    sq_a = a ** 2
    sum_sq_a = torch.sum(sq_a, dim=2).unsqueeze(2)  # m->[b, m, 1]
    sq_b = b ** 2
    sum_sq_b = torch.sum(sq_b, dim=2).unsqueeze(1)  # n->[b, 1, n]
    bt = b.permute(0, 2, 1)
    # return torch.sqrt(sum_sq_a + sum_sq_b - 2 * a.matmul(bt))
    return torch.sqrt(torch.abs(sum_sq_a + sum_sq_b - 2 * a.matmul(bt)) + 1e-8)
    # return torch.abs(sum_sq_a + sum_sq_b - 2 * a.matmul(bt))


def l1_loss(a, b):
    a = a.unsqueeze(1).expand(-1, 24, -1, -1, -1, -1)
    b = b.unsqueeze(2).expand(-1, -1, 24, -1, -1, -1)
    return torch.mean(torch.abs(a - b), dim=(3, 4, 5))


def operation(matrixs):
    H = torch.eye(matrixs.shape[1]).unsqueeze(0).expand(matrixs.shape[0], -1, -1) \
        - (1 / matrixs.shape[1]) * torch.ones(matrixs.shape)
    H = H.cuda()
    S = torch.square(matrixs)
    return -H.matmul(S).matmul(H) / 2


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # load data
    cfg.CONST.BATCH_SIZE = 40  # batch size must be 1 here
    cfg.CONST.N_VIEWS_RENDERING = 24
    cfg.TRAIN.UPDATE_N_VIEWS_RENDERING_PER_ITERATION = False
    cfg.TRAIN.UPDATE_N_VIEWS_RENDERING_PER_EPOCH = False
    train_data_loader, _ = pipe_train.load_data(cfg)

    # Set up networks
    cfg.CONST.WEIGHTS = inference_model_path
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    refiner = Refiner(cfg)
    merger = Merger(cfg)
    encoder, decoder, refiner, merger, _ = \
        pipe_test.setup_network(cfg, encoder, decoder, refiner, merger)

    mlp = MLP(cfg)
    mlp = mlp.cuda()
    mlp.apply(utils.helpers.init_weights)

    for p in encoder.module.parameters():
        p.requires_grad = False
    for p in decoder.module.parameters():
        p.requires_grad = False
    for p in merger.module.parameters():
        p.requires_grad = False
    for p in refiner.module.parameters():
        p.requires_grad = False
    
    # lr scheduler
    solver = torch.optim.Adam(mlp.parameters(),
                              lr=cfg.REDUCE.LEARNING_RATE, betas=cfg.TRAIN.BETAS)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(solver,
                                                        milestones=cfg.REDUCE.MILESTONES,
                                                        gamma=cfg.REDUCE.GAMMA)

    # Set up loss functions
    print('use l1 loss')
    loss_function = torch.nn.L1Loss(reduction='sum')

    print('training with score map')

    output_dir = os.path.join(results_dir, dt.now().isoformat())
    os.makedirs(output_dir)
    log_txt = open(os.path.join(output_dir, 'log.txt'), 'w')
    losses = []

    # Training loop
    for epoch_idx in range(0, cfg.REDUCE.NUM_EPOCHS):
        epoch_start_time = time()
        batch_end_time = time()
        
        batch_time = AverageMeter()
        loss_log = AverageMeter()
        
        mlp.train()
        n_batches = len(train_data_loader)
        
        for batch_idx, (_, _, rendering_images, _) in enumerate(train_data_loader):
            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images)

            # encoder_1
            with torch.no_grad():
                image_features = encoder.module.resnet_forward(rendering_images)

            # mlp
            feature = mlp(image_features).view(rendering_images.shape[0], rendering_images.shape[1], -1)
            
            # euclidean distances between the vectors
            vec_distance = euclidean_distances(feature, feature)
            diag = (torch.eye(vec_distance.shape[1]).repeat(vec_distance.shape[0], 1, 1).cuda() - 1) * -1
            vec_distance = vec_distance * diag
            
            # encoder_2
            with torch.no_grad():
                image_features = encoder.module.other_layers_forward(image_features, cfg.CONST.N_VIEWS_RENDERING)
            
            # decoder
            if cfg.REDUCE.LOSS_DATA == 1:
                with torch.no_grad():
                    _, generated_volumes = decoder(image_features, True)
                # dice distance between the volumes
                gt = dice_distance(generated_volumes, generated_volumes)
            elif cfg.REDUCE.LOSS_DATA == 2:
                with torch.no_grad():
                    raw_features, _ = decoder(image_features, False)
                    score_map = merger.module.calculate_score_map(cfg.CONST.N_VIEWS_RENDERING, raw_features)
                # dice distance between the volumes
                gt = l1_loss(score_map, score_map)
            
            gt = gt * diag

            # update
            mlp.zero_grad()
            if cfg.REDUCE.MATRIX_OPERATION:
                loss = loss_function(operation(vec_distance), operation(gt)) / \
                       (vec_distance.numel() - vec_distance.shape[0] * vec_distance.shape[1])
            else:
                loss = loss_function(vec_distance, gt) / \
                       (vec_distance.numel() - vec_distance.shape[0] * vec_distance.shape[1])
            loss.backward()
            solver.step()

            # Append loss to average metrics
            loss_log.update(loss.item())
            losses.append(loss.item())
            log_txt.write('epoch_%d - iter_%d - loss: %.8f\n' %
                          (epoch_idx + 1, batch_idx + 1, loss.item()))

            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            if batch_idx == 0 or (batch_idx + 1) % cfg.REDUCE.SHOW_TRAIN_STATE == 0:
                print('[Epoch %d/%d][Batch %d/%d] Batch_time: %.3f Learning_Rate: %f, Loss = %.8f'
                      % (epoch_idx + 1, cfg.REDUCE.NUM_EPOCHS, batch_idx + 1, n_batches,
                         batch_time.val,
                         lr_scheduler.optimizer.param_groups[0]['lr'], loss.item()))

        # Adjust learning rate
        lr_scheduler.step()
        epoch_end_time = time()
        print('[epoch %d] Avg_Loss = %.8f' % (epoch_idx + 1, loss_log.avg))
        log_txt.write('[epoch %d] Epoch_time: %.3f Avg_Loss = %.8f\n' %
                      (epoch_idx + 1, epoch_end_time - epoch_start_time, loss_log.avg))

        # Save weights to file
        if (epoch_idx + 1) % cfg.REDUCE.SAVE_FREQ == 0:
            torch.save(mlp, os.path.join(output_dir, 'epoch_' + str(epoch_idx + 1) + '.pth'))
            print('save pth file: epoch_' + str(epoch_idx + 1) + '.pth')

    log_txt.close()

    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.plot(list(range(len(losses))), losses, 'b--')
    plt.subplot(212)
    plt.plot(list(range(len(losses)))[int(len(losses) / 2):], losses[int(len(losses) / 2):], 'b--')
    plt.savefig(os.path.join(output_dir, 'loss.png'))


if __name__ == '__main__':
    main()
