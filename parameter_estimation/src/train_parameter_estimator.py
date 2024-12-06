import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import os
import glob
import parse
import argparse
import numpy as np # Sometimes DataLoaders are really slow if you import numpy first?
import pandas as pd
from time import time
from random import shuffle

from dataset import *
from parameter_estimator import ParameterEstimator
from parameter_estimator_convnext import ConvNextParameterEstimator

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class WeightedMSELoss(torch.nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', weights=None):
        super(WeightedMSELoss, self).__init__(size_average, reduce, reduction)
        self.weights = weights

    def forward(self, input, target):
        if self.weights is None:
            return F.mse_loss(input, target, reduction=self.reduction)
        else:
            return F.mse_loss(input * self.weights,
                              target * self.weights,
                              reduction=self.reduction)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='../../data/activity_assay/') 
    parser.add_argument('--frames_per_seq', type=int, default=8)
    parser.add_argument('--conv_size', type=int, default=32)
    parser.add_argument('--rnn_size', type=int, nargs='+', default=32)
    parser.add_argument('--fcnn_size', type=int, nargs='+', default=32)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--crop_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--validation_split', type=float, default=0.2)
    parser.add_argument('--num_test_datasets', type=int, default=0) #Reserve datasets unseen for testing in multi_parameter
    parser.add_argument('--save_name', type=str, default='../models/Test')
    parser.add_argument('--trial', action='store_true',
        help='Testing utility. Stop epoch after 1 batches')
    args = parser.parse_args()

    # GPU or CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Training on {device}')

    """
    Initialize all objects
    """
    # Dataset and data loaders
    transform = transforms.Compose([
        Sin2t(),
        RandomFlip(),
        RandomShift(),
        RandomRotation(),
        RandomCrop(args.crop_size),
        ToTensor(),
    ])
    paths = glob.glob(f'{args.directory}/*.hdf5')

    if args.num_test_datasets > 0:
        logger.info(f'Reserving {args.num_test_datasets} for testing')
        shuffle(paths)
        train_paths = paths[:len(paths) - args.num_test_datasets]
        test_paths = paths[len(paths) - args.num_test_datasets:]
    else:
        logger.info('Not using a separate test dataset')
        train_paths = paths
        test_paths = None

    #Concat dataset is so much easier than the build_dataframe nonsense I did before
    train_datasets = []
    for path in train_paths:
        train_datasets.append(NematicsSequenceDataset(path, transform=transform, frames_per_seq=args.frames_per_seq))
    output_dims = len(train_datasets[-1].attrs.keys())
    output_labels = sorted(list(train_datasets[-1].attrs.keys()))
    logger.info(f'Datasets have {output_dims} parameters: {output_labels}')
    train_datasets = torch.utils.data.ConcatDataset(train_datasets) 
    train_dataset, val_dataset = random_split(train_datasets, [0.8, 0.2])
    logger.info(f'Train dataset has size {len(train_dataset)}, Validation dataset has size {len(val_dataset)}')

    if test_paths is not None:
        test_datasets = []
        for path in test_paths:
            test_datasets.append(NematicsSequenceDataset(path, transform=transform, frames_per_seq=args.frames_per_seq))
        test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    else:
        logger.info('Reusing val dataset as test dataset')
        test_dataset = val_dataset

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # Model and optimizer
    model_kwargs = dict(
        input_size=args.crop_size,
        conv_size=args.conv_size,
        rnn_size=args.rnn_size,
        fcnn_size=args.fcnn_size,
        dropout=args.dropout,
        kernel_size=args.kernel_size,
        output_dims=output_dims,
    )

    # Original model
    # model = ParameterEstimator(**model_kwargs)
    # model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.92) # Seems rather aggressive

    # ConvNext model
    model = ConvNextParameterEstimator(**model_kwargs)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    if output_dims == 1:
        # Z dataset
        loss_func = torch.nn.MSELoss()
    else:
        # K, Z dataset: Median K ~ 0.13, Median Z ~ 0.025
        # Apply weights of 1., 5.2 (0.13 / 0.025 = 5.2)
        loss_func = WeightedMSELoss(weights=torch.FloatTensor([1., 5.2]).to(device))
        
    """ 
    Training loop begins here
    """
    logger.info("Starting to train")
    patient = 20
    loss_min = np.inf
    losses = []
    best_epoch = 0.

    for epoch in range(args.epochs):
        if epoch - best_epoch >= patient:
            logger.info('early stop at epoch %g' % best_epoch)
            logger.info('----------------------')
            break

        t = time()
        model.train()
        loss_train = 0.
        for batch in train_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            preds = model(inputs)
            targets = targets.to(device)

            loss = loss_func(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item() / len(train_loader)
            
            if args.trial: break

        model.eval()
        loss_val = 0.
        predictions = []
        actuals = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                preds = model(inputs)
                targets = targets.to(device)

                loss = loss_func(preds, targets)
                loss_val += loss / len(val_loader)

                predictions.append(preds.cpu().numpy())
                actuals.append(targets.cpu().numpy())

                if args.trial: break

        logger.info(f'Epoch {epoch}: train loss: {loss_train:.3g}, val_loss: {loss_val:.3g}\ttime={time()-t:.3g}')

        scheduler.step()
        losses.append(loss_val)
        if loss_val < loss_min:
            os.makedirs(args.save_name, exist_ok=True)
            # Save model weights
            torch.save({'state_dict': model.state_dict(),
                        'loss': loss_val,
                        'losses': losses},
                       f'{args.save_name}/model_weight.ckpt')

            # Save validation predictions
            predictions = np.concatenate(predictions, axis=0)
            actuals = np.concatenate(actuals, axis=0)
    
            df_pred = pd.DataFrame(predictions, columns=[ol+'_pred' for ol in output_labels])
            df_true = pd.DataFrame(actuals, columns=[ol+'_true' for ol in output_labels])
            df_val = pd.concat([df_true, df_pred], axis=1)
            df_val.to_csv(f'{args.save_name}/validation_predictions.csv')

            # Updat ebest epoch information
            best_epoch = epoch
            loss_min = loss_val
                
        if args.trial: 
            break

    """
    To make things more efficient, generate test predictions at the end of training instead of each epoch as before
    """
    logger.info('Loading best-performing checkpoint')
    info = torch.load(f'{args.save_name}/model_weight.ckpt', weights_only=True)
    model.load_state_dict(info['state_dict'])

    logger.info("Starting to predict")
    model.eval()
    with torch.no_grad():
        # Generate predictions on validation set (what we did the first time around)
        logger.info(f'Generating predictions on validation dataset with size {len(val_dataset)}')
        predictions = []
        actuals = []
    
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            preds = model(inputs)

            predictions.append(preds.cpu().numpy())
            actuals.append(targets.cpu().numpy())
            
            if args.trial: break

        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)

        df_pred = pd.DataFrame(predictions, columns=[ol+'_pred' for ol in output_labels])
        df_true = pd.DataFrame(actuals, columns=[ol+'_true' for ol in output_labels])
        df_val = pd.concat([df_true, df_pred], axis=1)
        df_val.to_csv(f'{args.save_name}/validation_predictions.csv')

        # Generate predictions on unseen parameter regime (the correct thing to do)
        logger.info(f'Generating predictions on testing dataset with size {len(test_dataset)}')
        predictions = []
        actuals = []
        
        for batch in test_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            preds = model(inputs)

            predictions.append(preds.cpu().numpy())
            actuals.append(targets.cpu().numpy())

            if args.trial: break

        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)

        df_pred = pd.DataFrame(predictions, columns=[ol+'_pred' for ol in output_labels])
        df_true = pd.DataFrame(actuals, columns=[ol+'_true' for ol in output_labels])
        df_test = pd.concat([df_true, df_pred], axis=1)
        df_test.to_csv(f'{args.save_name}/test_predictions.csv')        
