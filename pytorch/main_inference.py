# -*- coding: utf-8 -*-
"""
Created on 08/09/20

@author: Thomas Pellegrini
"""
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import time
import logging

import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import torch.utils.data

# from sklearn import metrics

from utilities import (create_folder, get_filename, create_logging, Mixup,
                       StatisticsContainer)
from models import (Cnn14, Cnn14_no_specaug, Cnn14_no_dropout,
                    Cnn6, Cnn10, ResNet22, ResNet22AuDeep, ResNet38, ResNet54, Cnn14_emb512, Cnn14_emb128,
                    Cnn14_emb32, MobileNetV1, MobileNetV1Audeep, MobileNetV2, LeeNet11, LeeNet24, DaiNet19,
                    Res1dNet31, Res1dNet51, Wavegram_Cnn14, Wavegram_Logmel_Cnn14,
                    Wavegram_Logmel128_Cnn14, Cnn14_16k, Cnn14_8k, Cnn14_mel32, Cnn14_mel128,
                    Cnn14_mixup_time_domain, Cnn14_DecisionLevelMax, Cnn14_DecisionLevelAtt)
from pytorch_utils import (move_data_to_device, count_parameters, count_flops,
                           do_mixup, forward)
from data_generator import (AudioSetDataset, TrainSampler, BalancedTrainSampler,
                            AlternateTrainSampler, EvaluateSampler, collate_fn)
from evaluate import Evaluator
import config
from losses import get_loss_func

# from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay

def read_class_labels(csv_path='metadata/class_labels_indices.csv'):
    with open(csv_path, "rt") as fh:
        num2str = {}
        first_line = True
        for ligne in fh:
            if first_line:
                first_line = False
                continue
            tab = ligne.rstrip().split(',')
            num2str[int(tab[0])] = tab[-1].replace('"', '')
        return num2str


def write_predictions_to_csv(clipwise_output, audio_names, prediction_outpath):
    # num2str = read_class_labels()
    num2str = config.ix_to_lb

    nb_files = len(audio_names)
    print("type", type(clipwise_output), nb_files)
    binary_predictions = np.argmax(clipwise_output, axis=-1)
    with open(prediction_outpath, "wt") as fh:
        fh.write("filename,prediction\n")
        for i, fid in enumerate(audio_names):
            fh.write("%s,%s\n" % (fid, num2str[binary_predictions[i]]))


# def plot_confusion_matrix(y_true, y_pred):
#     import matplotlib.pyplot as plt
#     from sklearn.metrics import confusion_matrix
#     cm = confusion_matrix(y_true, y_pred, normalize='true') # normalize=None
#     label_list = ["background", "chimpanze", "geunon", "mandrille", "redcap"]
#
#     np.set_printoptions(precision=2)
#     cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list)
#     disp = cmd.plot()
#     # fig.ax_.savefig("results/cm.png")
#     plt.savefig("results/cm1.png")

def plot_confusion_matrix(y_true, y_pred, uar):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    from itertools import product

    cm = confusion_matrix(y_true, y_pred, normalize='true')  # normalize=None
    # , labels=["background", "chimpanze", "geunon", "mandrille", "redcap"]
    classes_num = config.classes_num

    np.set_printoptions(precision=2)
    print(cm)
    label_list = ["B", "C", "G", "M", "R"]

    plt.figure(figsize=(7, 7))
    im_ = plt.imshow(cm, interpolation='nearest', cmap='viridis')
    plt.xticks(range(classes_num), label_list, rotation=0, fontsize=14)
    plt.yticks(range(classes_num), label_list, rotation=0, fontsize=14)
    plt.xlabel("Predicted label", fontsize=14)
    plt.ylabel("True label", fontsize=14)
    plt.title("UAR=%.1f%%" % (100. * uar), fontsize=14)

    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)
    # print text with appropriate color depending on background
    thresh = (cm.max() + cm.min()) / 2.0

    for i, j in product(range(classes_num), range(classes_num)):
        # print('%.2f'%cm[i,i])
        color = cmap_max if cm[i, j] < thresh else cmap_min
        plt.text(j, i, '%.1f%%' % (cm[i, j] * 100),
                 ha="center", va="center",
                 color=color, fontsize=14)
    plt.colorbar()
    plt.savefig("results/cm.png")


def save_emb_to_disk(emb_np, fpath):
    with open(fpath, 'wb') as f:
        np.save(f, emb_np)


def batch_inference(args):
    """Make predictions on a subset (tagging).

    Args:
      dataset_dir: str
      workspace: str
      data_type: 'balanced_train' | 'full_train'
      window_size: int
      hop_size: int
      mel_bins: int
      model_type: str
      loss_type: 'clip_bce'
      batch_size: int
      cuda: bool
    """

    # Arugments & parameters
    workspace = args.workspace
    data_type = args.data_type
    balanced = args.balanced
    augmentation = args.augmentation
    checkpoint_name = args.checkpoint_name
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    loss_type = args.loss_type
    batch_size = args.batch_size
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    subset = data_type

    use_audeep = True if args.audeep == 'yes' else False
    use_spec_augment = True if args.spec_augment == 'yes' else False
    use_cos_sched = True if args.use_cos_sched == 'yes' else False

    save_embeddings_to_disk = True if args.save_embeddings_to_disk == 'yes' else False

    num_workers = 8
    clip_samples = config.clip_samples
    classes_num = config.classes_num
    loss_func = get_loss_func(loss_type)

    # Paths
    black_list_csv = None
    if use_audeep:
        indexes_hdf5_path = os.path.join(workspace, 'hdf5s', 'indexes_audeep',
                                         '%s.h5' % (subset))
    else:
        indexes_hdf5_path = os.path.join(workspace, 'hdf5s', 'indexes',
                                         '%s.h5' % (subset))

    print("PATH === ", indexes_hdf5_path)
    # model_type + "_sauvegarde",

    checkpoints_dir = os.path.join(workspace, 'checkpoints/main',
                                   'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
                                       sample_rate, window_size, hop_size, mel_bins, fmin, fmax),
                                   'data_type=train', args.model_type,
                                  'loss_type={}'.format(args.loss_type), 'balanced={}'.format(args.balanced),
                                  'augmentation={}'.format(args.augmentation), 'spec_augment={}'.format(args.spec_augment),
                                  'signal_augmentation={}'.format(args.signal_augmentation),
                                  'audeep={}'.format(args.audeep), 'use_cos_sched={}'.format(args.use_cos_sched),
                                  'batch_size={}'.format(args.batch_size))

    # checkpoints_dir = os.path.join(workspace, 'checkpoints/main',
    #                                'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
    #                                    sample_rate, window_size, hop_size, mel_bins, fmin, fmax),
    #                                'data_type=train', model_type,
    #                                'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced),
    #                                'augmentation={}'.format(augmentation), 'spec_augment={}'.format(args.spec_augment),
    #                                'audeep={}'.format(args.audeep),
    #                                'batch_size={}'.format(batch_size))

    logs_dir = os.path.join(workspace, 'logs/main',
                            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
                                sample_rate, window_size, hop_size, mel_bins, fmin, fmax),
                            'data_type={}'.format(data_type), model_type,
                            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced),
                            'augmentation={}'.format(augmentation), 'spec_augment={}'.format(args.spec_augment),
                            'audeep={}'.format(args.audeep),
                            'batch_size={}'.format(batch_size))

    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
        device = 'cpu'

    # Model
    Model = eval(model_type)
    use_sigmoid = True if augmentation == 'mixup' else False
    print("%%%%%%  use_sigmoid", use_sigmoid)

    model = Model(sample_rate=sample_rate, window_size=window_size,
                  hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
                  classes_num=classes_num, use_sigmoid=use_sigmoid,
                  use_spec_augment=use_spec_augment)

    params_num = count_parameters(model)
    # flops_num = count_flops(model, clip_samples)
    logging.info('Parameters num: {}'.format(params_num))
    # logging.info('Flops num: {:.3f} G'.format(flops_num / 1e9))

    # Dataset will be used by DataLoader later. Dataset takes a meta as input
    # and return a waveform and a target.
    dataset = AudioSetDataset(sample_rate=sample_rate, use_audeep=use_audeep)

    sampler = EvaluateSampler(
        indexes_hdf5_path=indexes_hdf5_path, batch_size=batch_size)

    # Data loader
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_sampler=sampler, collate_fn=collate_fn,
                                         num_workers=num_workers, pin_memory=True)

    # Evaluator
    evaluator = Evaluator(model=model)

    # Resume training
    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
    out_dir = os.path.dirname(checkpoint_path)

    logging.info('Loading checkpoint {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    model.eval()

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)

    time1 = time.time()
    print("DEBUG, SUBSET", subset)

    if 'devel' in subset or 'train' in subset:
        if save_embeddings_to_disk:
            test_statistics, clipwise_output, target, audio_names, embeddings = evaluator.evaluate(loader,
                                                                                       use_audeep=use_audeep,
                                                                                       return_tagging_predictions_and_targets=True,
                                                                                        return_embeddings=True)
        else:
            test_statistics, clipwise_output, target, audio_names = evaluator.evaluate(loader,
                                                                                       use_audeep=use_audeep,
                                                                                       return_tagging_predictions_and_targets=True)

        target_outpath = os.path.join(out_dir, '%s_target.pth' % subset)
        torch.save(target, target_outpath)

        logging.info('Inference on {} mAP: {:.3f}'.format(subset,
                                                          np.mean(test_statistics['average_precision'])))

        logging.info('Inference on {} uar: {:.3f}'.format(subset,
                                                          np.mean(test_statistics['uar'])))

        plot_confusion_matrix(np.squeeze(target), np.argmax(clipwise_output, axis=-1), np.mean(test_statistics['uar']))

        logging.info("CM plot")

    elif 'test' in subset:

        output_dict = forward(
            model=model,
            generator=loader,
            use_audeep=use_audeep,
            return_target=True)
        clipwise_output = output_dict['clipwise_output']
        audio_names = output_dict['audio_name']
        embeddings = output_dict['embedding']

    prediction_outpath = os.path.join(out_dir, '%s_clipwise_predictions.pth' % subset)
    torch.save(clipwise_output, prediction_outpath)

    logging.info(
        'Predictions saved to: {}'
        ''.format(prediction_outpath))

    prediction_outpath = os.path.join(out_dir, '%s_clipwise_predictions.csv' % subset)
    write_predictions_to_csv(clipwise_output, audio_names, prediction_outpath)

    logging.info(
        'Predictions saved to: {}'
        ''.format(prediction_outpath))

    validate_time = time.time() - time1

    logging.info(
        'Inference time: {:.3f} s'
        ''.format(validate_time))

    save_embeddings_to_disk = True
    if save_embeddings_to_disk:
        save_emb_to_disk(embeddings, os.path.join(checkpoints_dir, "emb_%s.npy" % (subset)))

    logging.info('------------------------------------')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_eval = subparsers.add_parser('inference')
    parser_eval.add_argument('--workspace', type=str, required=True)
    parser_eval.add_argument('--data_type', type=str, default='devel', choices=['train', 'devel', 'test'])
    parser_eval.add_argument('--balanced', type=str, default='balanced', choices=['none', 'balanced', 'alternate'])
    parser_eval.add_argument('--augmentation', type=str, default='mixup', choices=['none', 'mixup'])
    parser_eval.add_argument('--spec_augment', type=str, default='yes', choices=['no', 'yes'])
    parser_eval.add_argument('--signal_augmentation', type=str, default='none', choices=['none', 'time_stretch', 'pitch_shift', 'both'])
    parser_eval.add_argument('--checkpoint_name', type=str, required=True)
    parser_eval.add_argument('--sample_rate', type=int, default=32000)
    parser_eval.add_argument('--window_size', type=int, default=1024)
    parser_eval.add_argument('--hop_size', type=int, default=320)
    parser_eval.add_argument('--mel_bins', type=int, default=64)
    parser_eval.add_argument('--fmin', type=int, default=50)
    parser_eval.add_argument('--fmax', type=int, default=14000)
    parser_eval.add_argument('--model_type', type=str, required=True)
    parser_eval.add_argument('--loss_type', type=str, default='clip_ce', choices=['clip_bce', 'clip_ce'])
    parser_eval.add_argument('--use_cos_sched', type=str, default='no', choices=['no', 'yes'])
    parser_eval.add_argument('--audeep', type=str, default='no', choices=['no', 'yes'])
    parser_eval.add_argument('--batch_size', type=int, default=32)
    parser_eval.add_argument('--save_embeddings_to_disk', type=str, default='no', choices=['no', 'yes'])
    parser_eval.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'inference':
        batch_inference(args)

    else:
        raise Exception('Error argument!')