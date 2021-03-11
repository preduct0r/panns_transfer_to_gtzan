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


    packed_hdf5_path = os.path.join(workspace, 'features_interspeech_final', 'interspeech_waveform_test.h5')
    create_folder(os.path.dirname(packed_hdf5_path))

    (audio_names, audio_paths) = traverse_folder(audios_dir)

    # audio_names = sorted(audio_names)
    # audio_paths = sorted(audio_paths)

    meta_train_df = pd.read_csv('/home/den/datasets/interspeech/lab/train.csv')
    meta_dev_df = pd.read_csv('/home/den/datasets/interspeech/lab/devel.csv')
    meta_test_df = pd.read_csv('/home/den/datasets/interspeech/lab/test.csv')

    audio_names_without_test, audio_names_test = [], []
    audio_paths_without_test, audio_paths_test = [], []
    for name,path in zip(audio_names,audio_paths):
        flag=False
        if 'test' not in name:
            audio_paths_without_test.append(path)
            audio_names_without_test.append(name)
            flag=True
        if not flag:
            audio_paths_test.append(path)
            audio_names_test.append(name)

    audio_names = audio_names_test
    audio_paths = audio_paths_test

    targets, folds = [], []
    for name in audio_names:
        if 'test' in name:
            targets.append(-1)
            folds.append('0')


    meta_dict = {
        'audio_name': audio_names,
        'audio_path': audio_paths,
        'target': targets,
        'fold': folds
    }

    audios_num = len(audio_names)

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
            print(n)
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