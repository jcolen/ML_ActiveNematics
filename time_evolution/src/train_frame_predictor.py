import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset import *
from res_ae import ResidualFramePredictor
from res_ae_convnext import ConvNextFramePredictor

import argparse
import os
import numpy as np
from time import time

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-z', '--activity', type=str, 
                        default=['z0', 'z0.001', 'z0.002', 'z0.005', 
                                 'z0.01', 'z0.015', 'z0.02', 'z0.025', 
                                 'z0.03', 'z0.035', 'z0.04', 'z0.05'])
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('--recurrent', type=str, default='ResidualNetwork')
    parser.add_argument('--channels', type=int, nargs='+', default=[2, 4, 6])
    parser.add_argument('--crop_size', type=int, default=48)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--freeze_recurrent', action='store_true')
    parser.add_argument('--freeze_spatial', action='store_true')
    parser.add_argument('--validation_split', type=float, default=0.2)
    parser.add_argument('--trial', action='store_true',
        help='Testing utility. Stop epoch after 1 batches')
    parser.add_argument('--save_name', type=str, default='../models/Test')
    args = parser.parse_args()

    #GPU or CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Training on {device}')
    
    # Dataset and data loaders
    transform = transforms.Compose([
        SinCos(),
        RandomFlip(0.5),
        RandomTranspose(0.5),
        RandomShift(0.5),
        RandomCrop(args.crop_size),
        ToTensor()])
    datasets = []
    for z in args.activity:
        datasets.append(NematicsSequenceDataset(f'../../data/activity_assay/{z}.hdf5', 
                                                frames_per_seq=10, #Lookahead of 3 for ConvNext
                                                #frames_per_seq=8, #Lookahead of 1 for ResAE
                                                transform=transform))
    dataset = torch.utils.data.ConcatDataset(datasets) #Concat dataset is so much easier than the build_dataframe nonsense I did before
    logger.info(f'Total dataset has size {len(dataset)}')
    train, test = random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # Model and optimizer
    #model = ResidualFramePredictor() # Original model
    model = ConvNextFramePredictor() # Updated model, more consistent
    model.to(device)
    
    if args.freeze_recurrent:	
        model.freeze_recurrent()
    if args.freeze_spatial: 
        model.freeze_spatial()

    # Residual Frame Predictor
    # optimizer = torch.optim.Adam(model.named_grad_parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.92)

    # ConvNext Frame Predictor - the more consistent one
    optimizer = torch.optim.AdamW(model.named_grad_parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)


    #Training
    patient = 20
    loss_min = np.Inf
    losses = []
    best_epoch = 0

    logger.info('Starting to train')
    
    for epoch in range(args.epochs):
        if epoch - best_epoch >= patient:
            logger.info('early stop at epoch %g' % best_epoch)
            break
        
        t = time()
        model.train()
        loss_train = 0.
        for batch in train_loader:
            batch = batch.to(device)
            inputs = batch[:, :7] # Lookback of 7 frames
            target = batch[:, 7:]
            preds = model(inputs, tmax=target.shape[1])
            loss = F.l1_loss(preds, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item() / len(train_loader)

            if args.trial: break

        model.eval()
        loss_test = 0.
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                inputs = batch[:, :7] #Lookback of 7 frames
                target = batch[:, 7:]
                preds = model(inputs, tmax=target.shape[1])
                loss = F.l1_loss(preds, target)
    
                loss_test += loss.item() / len(test_loader)

                if args.trial: break

        logger.info(f'Epoch {epoch}: train loss: {loss_train:.3g}, test loss: {loss_test:.3g}\ttime={time()-t:.3g}')

        scheduler.step()
        losses.append(loss_test)
        if loss_test < loss_min:
            os.makedirs(args.save_name, exist_ok=True)

            torch.save(
                {'state_dict': model.state_dict(),
                 'loss': loss_test,
                 'losses': losses},
                 f'{args.save_name}/model_weight.ckpt')
            best_epoch = epoch
            loss_min = loss_test

        if args.trial:
            break
