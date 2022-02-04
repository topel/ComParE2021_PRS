import numpy as np
import argparse
import csv
import os
import glob
import datetime
import time
import logging
import h5py
import librosa

from utilities import create_folder, get_sub_filepaths
import config


def create_indexes(args):
    """Create indexes a for dataloader to read for training. When users have
    a new task and their own data, they need to create similar indexes. The
    indexes contain meta information of "where to find the data for training".
    """

    # Arguments & parameters
    waveforms_hdf5_path = args.waveforms_hdf5_path
    indexes_hdf5_path = args.indexes_hdf5_path

    no_label = False
    if "test" in waveforms_hdf5_path:
        no_label = True

    # Paths
    create_folder(os.path.dirname(indexes_hdf5_path))

    with h5py.File(waveforms_hdf5_path, 'r') as hr:
        with h5py.File(indexes_hdf5_path, 'w') as hw:
            audios_num = len(hr['audio_name'])
            hw.create_dataset('audio_name', data=hr['audio_name'][:], dtype='S20')
            if not no_label:
                hw.create_dataset('target', data=hr['target'][:], dtype=np.bool)
            hw.create_dataset('hdf5_path', data=[waveforms_hdf5_path.encode()] * audios_num, dtype='S200')
            hw.create_dataset('index_in_hdf5', data=np.arange(audios_num), dtype=np.int32)

    print('Write to {}'.format(indexes_hdf5_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_create_indexes = subparsers.add_parser('create_indexes')
    parser_create_indexes.add_argument('--waveforms_hdf5_path', type=str, required=True,
                                       help='Path of packed waveforms hdf5.')
    parser_create_indexes.add_argument('--indexes_hdf5_path', type=str, required=True,
                                       help='Path to write out indexes hdf5.')

    parser_combine_full_indexes = subparsers.add_parser('combine_full_indexes')
    parser_combine_full_indexes.add_argument('--indexes_hdf5s_dir', type=str, required=True,
                                             help='Directory containing indexes hdf5s to be combined.')
    parser_combine_full_indexes.add_argument('--full_indexes_hdf5_path', type=str, required=True,
                                             help='Path to write out full indexes hdf5 file.')

    args = parser.parse_args()

    if args.mode == 'create_indexes':
        create_indexes(args)

    elif args.mode == 'combine_full_indexes':
        combine_full_indexes(args)

    else:
        raise Exception('Incorrect arguments!')