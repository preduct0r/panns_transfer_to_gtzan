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

import config, config_emocon
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



def pack_audio_files_to_hdf5_iemocap(args):
    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    mini_data = args.mini_data

    sample_rate = config_emocon.sample_rate
    clip_samples = config_emocon.clip_samples
    classes_num = config_emocon.classes_num
    lb_to_idx = config_emocon.lb_to_idx

    # Paths
    audios_dir = os.path.join(dataset_dir)

    if mini_data:
        packed_hdf5_path = os.path.join(workspace, 'features_emocon', 'minidata_waveform.h5')
    else:
        packed_hdf5_path = os.path.join(workspace, 'features_emocon', 'waveform.h5')
    create_folder(os.path.dirname(packed_hdf5_path))

    (audio_names, audio_paths) = traverse_folder(audios_dir)

    # audio_names = sorted(audio_names)
    # audio_paths = sorted(audio_paths)

    meta_df = pd.read_csv('/home/den/datasets/experiments/emocon_mult/meta.csv', sep=';')
    meta_train_df = pd.read_csv('/home/den/datasets/experiments/emocon_mult/meta_train.csv', sep=';')
    train_names = list(meta_train_df.cur_name)



    meta_dict = {
        'audio_name': np.array(audio_names),
        'audio_path': np.array(audio_paths),
        'target': np.array([int(meta_df[meta_df.cur_name==audio_name].arvalmix) for audio_name in audio_names]),
        'fold': np.array([0 if audio_name in train_names else 1 for audio_name in audio_names])}

    if mini_data:
        mini_num = 10
        total_num = len(meta_dict['audio_name'])
        random_state = np.random.RandomState(1234)
        indexes = random_state.choice(total_num, size=mini_num, replace=False)
        for key in meta_dict.keys():
            meta_dict[key] = meta_dict[key][indexes]

    audios_num = len(meta_dict['audio_name'])

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
    # subparsers = parser.add_subparsers(dest='mode')


    # Calculate feature for all audio files
    # parser_pack_audio = subparsers.add_parser('pack_audio_files_to_hdf5')
    # parser_pack_audio = subparsers.add_parser('pack_audio_files_to_hdf5_emocon')

    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')
    
    # Parse arguments
    args = parser.parse_args()
    

    pack_audio_files_to_hdf5_iemocap(args)


    # raise Exception('Incorrect arguments!')