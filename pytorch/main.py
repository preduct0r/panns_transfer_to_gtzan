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


    for i in range(3,4):
        # try:
            hdf5_path = os.path.join(workspace, 'features_interspeech_march', 'interspeech_waveform_{}.h5'.format(i))

            checkpoints_dir = os.path.join(workspace, 'checkpoints', 'interspeech_march', 'augmentation={}'.format(augmentation))

            create_folder(checkpoints_dir)

            statistics_path = os.path.join(workspace, 'statistics', filename,
                'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain),
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


            start = time.time()
            time_passed = 0
            best_recall = 0.7
            while time_passed < 5400:
                # Model
                Model = eval(model_type)

                # TODO захардкодил classes num- это нехорошо
                model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                              3, freeze_base)

                # Statistics
                statistics_container = StatisticsContainer(statistics_path)

                if pretrain:
                    logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
                    model.load_from_pretrain(pretrained_checkpoint_path)

                iteration = 0

                # Parallel
                print('GPU number: {}'.format(torch.cuda.device_count()))
                model = torch.nn.DataParallel(model)

                if 'cuda' in device:
                    model.to(device)

                # Evaluator
                evaluator = Evaluator(model=model)

                # Optimizer
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                                       eps=1e-09, weight_decay=0., amsgrad=True)

                num = random.randint(0, 1000000000)
                torch.manual_seed(num)
                # Train on mini batches
                for batch_data_dict in train_loader:

                    # import crash
                    # asdf
                    torch.cuda.empty_cache()
                    # Evaluate
                    if iteration % 100 == 0 and iteration > 0:
                        if resume_iteration > 0 and iteration == resume_iteration:
                            pass
                        else:
                            logging.info('------------------------------------')
                            logging.info('Iteration: {}'.format(iteration))

                            train_fin_time = time.time()

                        statistics, _ = evaluator.evaluate(validate_loader)
                        logging.info('Validate precision: {:.3f}'.format(statistics['precision']))
                        logging.info('Validate recall: {:.3f}'.format(statistics['recall']))
                        logging.info('Validate f_score: {:.3f}'.format(statistics['f_score']))
                        logging.info('\n'+ str(statistics['cm']))

                        # statistics_container.append(iteration, statistics, 'validate')
                        # statistics_container.dump()

                        # train_time = train_fin_time - train_bgn_time
                        # validate_time = time.time() - train_fin_time
                        #
                        # logging.info(
                        #     'Train time: {:.3f} s, validate time: {:.3f} s'
                        #     ''.format(train_time, validate_time))
                        #
                        # train_bgn_time = time.time()

                            # Save model
                        if iteration >= 1000 and statistics['recall'] > best_recall:
                            best_recall = statistics['recall']
                            checkpoint = {
                                'iteration': iteration,
                                'model': model.module.state_dict()}

                            checkpoint_path = os.path.join(
                                checkpoints_dir, 'model_{}.pth'.format(i))

                            torch.save(checkpoint, checkpoint_path)
                            logging.info('Model saved to {}'.format(checkpoint_path))

                            _, output_dict = evaluator.evaluate(validate_loader)

                            with open('/home/den/Documents/random_search_march.txt', 'a') as f:
                                f.write('FOLD: {}\n'.format(str(i)))
                                f.write('Current time: {}\n'.format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
                                f.write('recall: {}\n'.format(statistics['recall']))
                                f.write('precision: {}\n'.format(statistics['precision']))
                                f.write('fscore: {}\n'.format(statistics['f_score']))
                                f.write('seed: {}\n'.format(num))
                                f.write('iteration: {}\n\n\n'.format(iteration))


                            df = pd.DataFrame(columns=['filename', 'label', 0, 1, 2])
                            df.loc[:, 'filename'] = output_dict['audio_name']
                            df.loc[:, 'label'] = output_dict['target']

                            df.loc[:, [0, 1, 2]] = np.vstack(output_dict['clipwise_output2'])
                            df.to_csv('/home/den/Documents/devs_march/df_dev_prob_{}.csv'.format(str(i)), index=False, sep=',')



                    if 'mixup' in augmentation:
                        batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(len(batch_data_dict['waveform']))

                    # Move data to GPU
                    for key in batch_data_dict.keys():
                        batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

                    # Train
                    model.train()

                    if 'mixup' in augmentation:
                        batch_output_dict = model(batch_data_dict['waveform'],
                            batch_data_dict['mixup_lambda'])
                        """{'clipwise_output': (batch_size, classes_num), ...}"""

                        batch_target_dict = {'target': do_mixup(batch_data_dict['target'],
                            batch_data_dict['mixup_lambda'])}
                        """{'target': (batch_size, classes_num)}"""
                    else:
                        batch_output_dict = model(batch_data_dict['waveform'], None)
                        """{'clipwise_output': (batch_size, classes_num), ...}"""

                        batch_target_dict = {'target': batch_data_dict['target']}
                        """{'target': (batch_size, classes_num)}"""

                    # loss
                    loss = loss_func(batch_output_dict, batch_target_dict)
                    # print(iteration, loss)

                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Stop learning
                    if iteration == stop_iteration:
                        break

                    iteration += 1
                    time_passed = time.time() - start

        # except:
        #     pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_train.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_train.add_argument('--holdout_fold', type=str, choices=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], required=True)
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