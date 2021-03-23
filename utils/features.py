import os
import sys
import numpy as np
import argparse
import h5py
import librosa
import matplotlib.pyplot as plt
import time
import csv
import math
import re
import random
import pandas as pd
from glob import glob

import config, config_emocon, config_interspeech
from utilities import create_folder, traverse_folder, float32_to_int16


def to_one_hot(k, classes_num):
    target = np.zeros(classes_num)
    target[k] = 1
    return target


def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]



def pack_audio_files_to_hdf5_interspeech(args):
    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    mini_data = args.mini_data

    sample_rate = config_interspeech.sample_rate
    clip_samples = config_interspeech.clip_samples
    classes_num = config_interspeech.classes_num

    # Paths
    audios_dir = os.path.join(dataset_dir, 'wav')
    (audio_names, audio_paths) = traverse_folder(audios_dir)

    meta_train_df = pd.read_csv('/home/den/datasets/interspeech/lab/train.csv')
    meta_dev_df = pd.read_csv('/home/den/datasets/interspeech/lab/devel.csv')
    meta_df = meta_train_df.append(meta_dev_df, ignore_index=True)

    audio_names_without_test = []
    audio_paths_without_test = []
    for name, path in zip(audio_names, audio_paths):
        if 'test' not in name:
            audio_paths_without_test.append(path)
            audio_names_without_test.append(name)

    audio_names = audio_names_without_test
    audio_paths = audio_paths_without_test
    idxs = list(range(len(audio_names)))




    for ii, fold_file in enumerate(glob('/home/den/workspaces/panns/folds_from_nastya/*.csv')):

        packed_hdf5_path = os.path.join(workspace, 'features_interspeech_march', 'interspeech_waveform_{}.h5'.format(ii))
        create_folder(os.path.dirname(packed_hdf5_path))

        cur_audio_names, cur_audio_paths, targets, folds = [], [], [], []
        val_df = pd.read_csv(fold_file).loc[:,['filename', 'label']]
        val_df['filename'] = [x +'.wav' for x in val_df['filename']]
        temp = meta_df.loc[:,['filename', 'label']]
        train_df = pd.concat([temp, val_df]).drop_duplicates(keep=False)



        for _, row in train_df.iterrows():
            targets.append(int(row[-1]))
            cur_audio_names.append(row[-2])
            idx = audio_names.index(row[-2])
            cur_audio_paths.append(audio_paths[idx])
            folds.append('1')

        for _, row in val_df.iterrows():
            targets.append(int(row[-1]))
            cur_audio_names.append(row[-2])
            idx = audio_names.index(row[-2])
            cur_audio_paths.append(audio_paths[idx])
            folds.append('0')



        meta_dict = {
            'audio_name': cur_audio_names,
            'audio_path': cur_audio_paths,
            'target': targets,
            'fold': folds
        }

        audios_num = len(cur_audio_names)

        feature_time = time.time()
        with h5py.File(packed_hdf5_path, 'w') as hf:
            hf.create_dataset(
                name='audio_name',
                shape=(audios_num,),
                dtype='S80')

            hf.create_dataset(
                name='waveform',
                shape=(audios_num, clip_samples),
                dtype=np.int16)

            hf.create_dataset(
                name='target',
                shape=(audios_num, classes_num),
                dtype=np.float32)

            hf.create_dataset(
                name='fold',
                shape=(audios_num,),
                dtype=np.int32)

            for n in range(audios_num):
                # print(n)
                audio_name = meta_dict['audio_name'][n]
                fold = meta_dict['fold'][n]
                audio_path = meta_dict['audio_path'][n]
                (audio, fs) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

                audio = pad_truncate_sequence(audio, clip_samples)

                hf['audio_name'][n] = audio_name.encode()
                hf['waveform'][n] = float32_to_int16(audio)
                hf['target'][n] = to_one_hot(meta_dict['target'][n], classes_num)
                hf['fold'][n] = meta_dict['fold'][n]

        print('Write hdf5 to {}'.format(packed_hdf5_path))
        print('Time: {:.3f} s'.format(time.time() - feature_time))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')


    # Calculate feature for all audio files
    parser_pack_audio = subparsers.add_parser('pack_audio_files_to_hdf5')
    parser_pack_audio = subparsers.add_parser('pack_audio_files_to_hdf5_interspeech')

    parser_pack_audio.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_pack_audio.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_pack_audio.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.mode == 'pack_audio_files_to_hdf5':
        pack_audio_files_to_hdf5(args)
    elif args.mode == 'pack_audio_files_to_hdf5_interspeech':
        pack_audio_files_to_hdf5_interspeech(args)
    else:
        raise Exception('Incorrect arguments!')