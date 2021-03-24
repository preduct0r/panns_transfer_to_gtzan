import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import time
import logging
import matplotlib.pyplot as plt
import pandas as pd
import random
from time import gmtime, strftime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from config import (sample_rate, classes_num, mel_bins, fmin, fmax, window_size,
                    hop_size, window, pad_mode, center, ref, amin, top_db)
from losses import get_loss_func
from pytorch_utils import move_data_to_device, do_mixup
from utilities import (create_folder, get_filename, create_logging, StatisticsContainer, Mixup)
from data_generator import GtzanDataset, TrainSampler, EvaluateSampler, collate_fn, BalancedBatchSampler
from models import Transfer_Cnn14
from evaluate import Evaluator

# для воспроизводимости результатов
# random.seed(0)
np.random.seed(0)
torch.manual_seed(500)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(24)


def train(args):
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    freeze_base = args.freeze_base
    loss_type = args.loss_type
    augmentation = args.augmentation
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    resume_iteration = args.resume_iteration
    stop_iteration = args.stop_iteration
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    filename = args.filename
    num_workers = 8

    loss_func = get_loss_func(loss_type)
    pretrain = True if pretrained_checkpoint_path else False

    for i in range(1, 5):
        # try:
        hdf5_path = os.path.join(workspace, 'features_interspeech_march', 'interspeech_waveform_{}.h5'.format(i))

        checkpoints_dir = os.path.join(workspace, 'checkpoints', 'interspeech_march',
                                       'augmentation={}'.format(augmentation))

        create_folder(checkpoints_dir)

        statistics_path = os.path.join(workspace, 'statistics', filename,
                                       'holdout_fold={}'.format(holdout_fold), model_type,
                                       'pretrain={}'.format(pretrain),
                                       'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation),
                                       'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base),
                                       'statistics.pickle')
        create_folder(os.path.dirname(statistics_path))

        logs_dir = os.path.join(workspace, 'logs', filename,
                                'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain),
                                'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation),
                                'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base))
        create_logging(logs_dir, 'w')
        logging.info(args)

        if 'cuda' in device:
            logging.info('Using GPU.')
        else:
            logging.info('Using CPU. Set --cuda flag to use GPU.')

        dataset = GtzanDataset()

        # Data generator
        train_sampler = TrainSampler(
            hdf5_path=hdf5_path,
            holdout_fold=holdout_fold,
            batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size)

        validate_sampler = EvaluateSampler(
            hdf5_path=hdf5_path,
            holdout_fold=holdout_fold,
            batch_size=batch_size)

        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_sampler=train_sampler, collate_fn=collate_fn,
                                                   num_workers=num_workers, pin_memory=True)

        validate_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_sampler=validate_sampler, collate_fn=collate_fn,
                                                      num_workers=num_workers, pin_memory=True)

        if 'mixup' in augmentation:
            mixup_augmenter = Mixup(mixup_alpha=1.)


        # Model
        Model = eval(model_type)

        # TODO захардкодил classes num- это нехорошо
        model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                      3, freeze_base)

        # Statistics
        statistics_container = StatisticsContainer(statistics_path)



        pretrained_checkpoint_path = '/home/den/workspaces/panns/checkpoints/interspeech_march/augmentation=mixup/model_{}.pth'.format(i)
        logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))


        # Model
        Model = eval(model_type)

        model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                      3, freeze_base)

        model.load_from_pretrain(pretrained_checkpoint_path)

        # Parallel
        model = torch.nn.DataParallel(model)

        if 'cuda' in device:
            model.to(device)

        # Evaluator
        evaluator = Evaluator(model=model)

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                               eps=1e-09, weight_decay=0., amsgrad=True)


        # Evaluator
        evaluator = Evaluator(model=model)

        _, output_dict = evaluator.evaluate(validate_loader)


        df = pd.DataFrame(columns=['filename', 'label', 0, 1, 2])
        df.loc[:, 'filename'] = output_dict['audio_name']

        df.loc[:, [0, 1, 2]] = np.vstack(output_dict['clipwise_output2'])


        df.loc[:, 'label'] = np.argmax(np.vstack(output_dict['clipwise_output2']), axis=1)

        df.to_csv('/home/den/Documents/dev_again/df_dev_prob_{}.csv'.format(str(i)), index=False,
                              sep=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_train.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_train.add_argument('--holdout_fold', type=str,
                              choices=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--pretrained_checkpoint_path', type=str)
    parser_train.add_argument('--freeze_base', action='store_true', default=False)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--augmentation', type=str, choices=['none', 'mixup'], required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--resume_iteration', type=int)
    parser_train.add_argument('--stop_iteration', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)

    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')