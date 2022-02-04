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

from utilities import (create_folder, get_filename, create_logging,
    float32_to_int16, pad_or_truncate, read_metadata)
import config

def load_audeep(fpath, skip_first_line=False, start_ind=2, end_ind=-2, has_labels=True):
    features = {}
    # labels = []
    # fids = []
    ind = 0
    with open(fpath, "rt") as fh:
        for ligne in fh:
            if skip_first_line:
                skip_first_line = False
                continue
            tab = ligne.rstrip().split(',')
            features[tab[0]] = np.array([float(el) for el in tab[start_ind:end_ind]], dtype=np.float32)
            # if ind < 1: print(features[tab[0]])
            # if has_labels:
            #     labels.append(int(tab[-1]))
            # fids.append(tab[0])
            ind += 1
    return features



def pack_waveforms_to_hdf5(args):
    """Pack waveform and target of several audio clips to a single hdf5 file.
    This can speed up loading and training.
    """

    # Arguments & parameters
    audios_dir = args.audios_dir
    csv_path = args.csv_path
    waveforms_hdf5_path = args.waveforms_hdf5_path
    mini_data = args.mini_data
    audeep_fpath = args.audeep_fpath

    no_label = False
    if "test" in csv_path:
        no_label = True

    clip_samples = config.clip_samples
    classes_num = config.classes_num
    sample_rate = config.sample_rate
    # id_to_ix = config.id_to_ix
    lb_to_ix = config.lb_to_ix
    audeep_dim = config.audeep_dim
    print(classes_num)


    # Paths
    if mini_data:
        prefix = 'mini_'
        waveforms_hdf5_path += '.mini'
    else:
        prefix = ''

    create_folder(os.path.dirname(waveforms_hdf5_path))

    logs_dir = '_logs/pack_waveforms_to_hdf5/{}{}'.format(prefix, get_filename(csv_path))
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')
    logging.info('Write logs to {}'.format(logs_dir))

    # Read csv file
    meta_dict = read_metadata(csv_path, classes_num, lb_to_ix)

    if mini_data:
        mini_num = 10
        for key in meta_dict.keys():
            meta_dict[key] = meta_dict[key][0: mini_num]

    audios_num = len(meta_dict['audio_name'])

    # Pack waveform to hdf5
    total_time = time.time()

    with h5py.File(waveforms_hdf5_path, 'w') as hf:
        hf.create_dataset('audio_name', shape=((audios_num,)), dtype='S20')
        hf.create_dataset('waveform', shape=((audios_num, clip_samples)), dtype=np.int16)
        if not no_label:
            hf.create_dataset('target', shape=((audios_num, classes_num)), dtype=np.bool)
        if audeep_fpath is not None:
            hf.create_dataset('audeep', shape=((audios_num, audeep_dim)), dtype=np.float32)
            has_labels = not no_label
            audeep_dict = load_audeep(audeep_fpath, skip_first_line=True, start_ind=2, end_ind=-2, has_labels=has_labels)
        hf.attrs.create('sample_rate', data=sample_rate, dtype=np.int32)
        
        
        # Pack waveform & target of several audio clips to a single hdf5 file
        for n in range(audios_num):
            audio_path = os.path.join(audios_dir, meta_dict['audio_name'][n])

            if os.path.isfile(audio_path):
                logging.info('{} {}'.format(n, audio_path))
                (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
                audio = pad_or_truncate(audio, clip_samples)

                hf['audio_name'][n] = meta_dict['audio_name'][n].encode()
                hf['waveform'][n] = float32_to_int16(audio)
                if audeep_fpath is not None:
                    if n<1: print(meta_dict['audio_name'][n], audeep_dict[meta_dict['audio_name'][n]].shape)
                    hf['audeep'][n] = audeep_dict[meta_dict['audio_name'][n]]
                if not no_label:
                    hf['target'][n] = meta_dict['target'][n]
            else:
                logging.info('{} File does not exist! {}'.format(n, audio_path))

    logging.info('Write to {}'.format(waveforms_hdf5_path))
    logging.info('Pack hdf5 time: {:.3f}'.format(time.time() - total_time))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_pack_wavs = subparsers.add_parser('pack_waveforms_to_hdf5')
    parser_pack_wavs.add_argument('--csv_path', type=str, required=True,
                                  help='Path of csv file containing audio info to be downloaded.')
    parser_pack_wavs.add_argument('--audios_dir', type=str, required=True,
                                  help='Directory to save out downloaded audio.')
    parser_pack_wavs.add_argument('--waveforms_hdf5_path', type=str, required=True,
                                  help='Path to save out packed hdf5.')
    parser_pack_wavs.add_argument('--mini_data', action='store_true', default=False,
                                  help='Set true to only download 10 audios for debugging.')
    parser_pack_wavs.add_argument('--audeep_fpath', type=str, required=False,
                                  help='CSV file path with audeep features.')

    args = parser.parse_args()

    if args.mode == 'pack_waveforms_to_hdf5':
        pack_waveforms_to_hdf5(args)
    else:
        raise Exception('Incorrect arguments!')