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


def pack_audio_files_to_hdf5(args):

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    mini_data = args.mini_data

    sample_rate = config.sample_rate
    clip_samples = config.clip_samples
    classes_num = config.classes_num
    lb_to_idx = config.lb_to_idx

    # Paths
    audios_dir = os.path.join(dataset_dir)

    if mini_data:
        packed_hdf5_path = os.path.join(workspace, 'features', 'minidata_waveform.h5')
    else:
        packed_hdf5_path = os.path.join(workspace, 'features', 'waveform.h5')
    create_folder(os.path.dirname(packed_hdf5_path))

    (audio_names, audio_paths) = traverse_folder(audios_dir)
    
    audio_names = sorted(audio_names)
    audio_paths = sorted(audio_paths)

    meta_dict = {
        'audio_name': np.array(audio_names), 
        'audio_path': np.array(audio_paths), 
        'target': np.array([lb_to_idx[audio_name.split('.')[0]] for audio_name in audio_names]), 
        'fold': np.arange(len(audio_names)) % 10 + 1}
    
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


def pack_audio_files_to_hdf5_emocon(args):
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
        packed_hdf5_path = os.path.join(workspace, 'features_emocon', 'emocon_emo_waveform.h5')
    create_folder(os.path.dirname(packed_hdf5_path))

    (audio_names, audio_paths) = traverse_folder(audios_dir)

    # audio_names = sorted(audio_names)
    # audio_paths = sorted(audio_paths)

    meta_df = pd.read_csv('/home/den/datasets/emocon/meta.csv', sep=';')
    temp_dict = {'sad': 'sad', 'happy': 'hap', 'angry': 'ang'}
    meta_df['cur_label'] = [temp_dict.get(x) for x in meta_df.cur_label]
    meta_train_df = pd.read_csv('/home/den/datasets/emocon/meta_train.csv', sep=';')
    meta_train_df['cur_label'] = [temp_dict.get(x) for x in meta_train_df.cur_label]
    train_names = list(meta_train_df.cur_name)

    meta_emo_df = pd.read_csv('/home/den/Documents/meta_emo_for_panns_emocon.csv', sep=';')
    idxs = []
    for idx, i in enumerate(audio_names):
        if i in meta_emo_df.cur_name.values:
            idxs.append(idx)
    audio_names = np.array([audio_names[x] for x in idxs])
    audio_paths = np.array([audio_paths[x] for x in idxs])

    idxs = []
    for idx, i in enumerate(train_names):
        if i in meta_emo_df.cur_name.values:
            idxs.append(idx)
    train_names = np.array([train_names[x] for x in idxs])



    meta_dict = {
        'audio_name': audio_names,
        'audio_path': audio_paths,
        'target': np.array([lb_to_idx.get(meta_df[meta_df.cur_name==audio_name].cur_label.values[0]) for audio_name in audio_names]),
        'fold': np.arange(len(audio_names)) % 10 + 1}

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
    subparsers = parser.add_subparsers(dest='mode')


    # Calculate feature for all audio files
    parser_pack_audio = subparsers.add_parser('pack_audio_files_to_hdf5')
    parser_pack_audio = subparsers.add_parser('pack_audio_files_to_hdf5_emocon')

    parser_pack_audio.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_pack_audio.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_pack_audio.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.mode == 'pack_audio_files_to_hdf5':
        pack_audio_files_to_hdf5(args)
    elif args.mode == 'pack_audio_files_to_hdf5_emocon':
        pack_audio_files_to_hdf5_emocon(args)
    else:
        raise Exception('Incorrect arguments!')