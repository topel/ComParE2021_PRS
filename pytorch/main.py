import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import time
import logging

import torch
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import LambdaLR

from ema import ExponentialMovingAverage

import copy

from utilities import (create_folder, get_filename, create_logging, Mixup,
                       StatisticsContainer)
from models import (Cnn14, Cnn14_no_specaug, Cnn14_no_dropout,
                    Cnn6, Cnn10, ResNet22, Transfer_ResNet22, ResNet22AuDeep, ResNet38, ResNet22Cnn10, ResNet22_mixup_time_domain, ResNet22_mixup_time_freq, ResNet54, Cnn14_emb512, Cnn14_emb128,
                    Cnn14_emb32, MobileNetV1, MobileNetV1Audeep, MobileNetV2, TransEnc, LeeNet11, LeeNet24, DaiNet19,
                    Res1dNet31, Res1dNet51, WideResNet28, Wavegram_Cnn14, Wavegram_Logmel_Cnn14,
                    Wavegram_Logmel128_Cnn14, Cnn14_16k, Cnn14_8k, Cnn14_mel32, Cnn14_mel128,
                    Cnn14_mixup_time_domain, Cnn14_DecisionLevelMax, Cnn14_DecisionLevelAtt)
from pytorch_utils import (move_data_to_device, count_parameters, count_flops,
                           do_mixup)
from data_generator import (AudioSetDataset, TrainSampler, BalancedTrainSampler,
                            AlternateTrainSampler, EvaluateSampler, collate_fn)
from evaluate import Evaluator
import config
from losses import get_loss_func

import socket

host_name = socket.gethostname()
print(host_name)

# if host_name == 'erebor':
from tensorboardX import SummaryWriter



def dir_path_suffix(args, pretrain=False):
        if pretrain:
            suffix = os.path.join(args.filename,\
            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
                args.sample_rate, args.window_size, args.hop_size, args.mel_bins, args.fmin, args.fmax),
                                  'data_type={}'.format(args.data_type), args.model_type,
                                  'pretrain={}'.format(pretrain), 'freeze_base={}'.format(args.freeze_base),
                                  'loss_type={}'.format(args.loss_type), 'balanced={}'.format(args.balanced),
                                  'augmentation={}'.format(args.augmentation), 'spec_augment={}'.format(args.spec_augment),
                                  'audeep={}'.format(args.audeep), 'use_cos_sched={}'.format(args.use_cos_sched),
                                  'batch_size={}'.format(args.batch_size))
        else:
            suffix = os.path.join(args.filename,\
            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
                args.sample_rate, args.window_size, args.hop_size, args.mel_bins, args.fmin, args.fmax),
                                  'data_type={}'.format(args.data_type), args.model_type,
                                  'loss_type={}'.format(args.loss_type), 'balanced={}'.format(args.balanced),
                                  'augmentation={}'.format(args.augmentation), 'spec_augment={}'.format(args.spec_augment),
                                  'signal_augmentation={}'.format(args.signal_augmentation),
                                  'audeep={}'.format(args.audeep), 'use_cos_sched={}'.format(args.use_cos_sched),
                                  'batch_size={}'.format(args.batch_size))
        return suffix

def train(args):
    """Train AudioSet tagging model.

    Args:
      dataset_dir: str
      workspace: str
      data_type: 'balanced_train' | 'full_train'
      window_size: int
      hop_size: int
      mel_bins: int
      model_type: str
      loss_type: 'clip_bce'
      balanced: 'none' | 'balanced' | 'alternate'
      augmentation: 'none' | 'mixup'
      batch_size: int
      learning_rate: float
      resume_iteration: int
      early_stop: int
      accumulation_steps: int
      cuda: bool
    """

    # Arugments & parameters
    workspace = args.workspace
    data_type = args.data_type
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    freeze_base = args.freeze_base
    loss_type = args.loss_type
    balanced = args.balanced
    augmentation = args.augmentation
    signal_augmentation = args.signal_augmentation

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    filename = args.filename

    use_audeep = True if args.audeep == 'yes' else False
    use_spec_augment = True if args.spec_augment == 'yes' else False
    use_time_stretch = True if signal_augmentation == 'time_stretch' or signal_augmentation == 'both' else False
    use_pitch_shift = True if signal_augmentation == 'pitch_shift' or signal_augmentation == 'both' else False
    use_cos_sched = True if args.use_cos_sched == 'yes' else False
    do_binary_task = True if args.task == 'binary' else False
    use_one_hot = True if augmentation == 'mixup' or do_binary_task else False

    num_workers = 8
    clip_samples = config.clip_samples
    classes_num = config.classes_num
    loss_func = get_loss_func(loss_type)
    pretrain = True if pretrained_checkpoint_path else False

    # Paths
    black_list_csv = None

    if use_audeep:
        train_indexes_hdf5_path = os.path.join(workspace, 'hdf5s', 'indexes_audeep',
                                               '{}.h5'.format(data_type))
        eval_bal_indexes_hdf5_path = os.path.join(workspace,
                                                  'hdf5s', 'indexes_audeep', 'devel.h5')
    else:
        train_indexes_hdf5_path = os.path.join(workspace, 'hdf5s', 'indexes',
                                               '{}.h5'.format(data_type))
        eval_bal_indexes_hdf5_path = os.path.join(workspace,
                                                  'hdf5s', 'indexes', 'devel.h5')

    # eval_test_indexes_hdf5_path = os.path.join(workspace, 'hdf5s', 'indexes',
    #                                            'test.h5')

    # checkpoints_dir = os.path.join(workspace, 'checkpoints', dir_path_suffix(args, pretrain=pretrain), '16k_specAug8')
    checkpoints_dir = os.path.join(workspace, 'checkpoints', dir_path_suffix(args, pretrain=pretrain))
    create_folder(checkpoints_dir)

    # statistics_path = os.path.join(workspace, 'statistics', dir_path_suffix(args, pretrain=pretrain), '16k_specAug8', 'statistics.pkl')
    statistics_path = os.path.join(workspace, 'statistics', dir_path_suffix(args, pretrain=pretrain), 'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', dir_path_suffix(args, pretrain=pretrain) )

    # if host_name == 'erebor':
    writer = SummaryWriter(logs_dir)

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

    if model_type == 'TransEnc':
        N = 6  # nb of layers, paper value: 6
        d_model = 128
        d_ff = 128  # 2048 dim of the feed-forward layer
        h = 8  # nb of attention heads

        # d_model = 512
        # d_ff = 2048  # 2048 dim of the feed-forward layer
        # h = 16  # nb of attention heads

        attn_dropout = 0.5
        dropout = 0.5
        enc_dropout=0.5

        from transformer_utils import EncoderLayer, MultiHeadedAttention, \
            PositionwiseFeedForward

        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout=attn_dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        layer = EncoderLayer(d_model, c(attn), c(ff), enc_dropout)

        model = Model(sample_rate=sample_rate, window_size=window_size,
                      hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
                      classes_num=classes_num,  N=N, layer=layer, d_model=d_model,
                      freeze_base=freeze_base, use_sigmoid=use_one_hot,
                      use_spec_augment=use_spec_augment, use_time_stretch=use_time_stretch,
                      use_pitch_shift=use_pitch_shift)

    else:
        model = Model(sample_rate=sample_rate, window_size=window_size,
                      hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
                      classes_num=classes_num, freeze_base=freeze_base, use_sigmoid=use_one_hot,
                      use_spec_augment=use_spec_augment, use_time_stretch=use_time_stretch,
                      use_pitch_shift=use_pitch_shift, do_binary_task=do_binary_task)

    # print(model)

    # exp. moving average of weights
    ema = ExponentialMovingAverage(model.named_parameters(), decay=0.995, device=device)

    params_num_learnable, params_num = count_parameters(model)
    # flops_num = count_flops(model, clip_samples)
    logging.info('Parameters learnable_num {}, total num {}'.format(params_num_learnable, params_num))
    # logging.info('Flops num: {:.3f} G'.format(flops_num / 1e9))

    # Dataset will be used by DataLoader later. Dataset takes a meta as input
    # and return a waveform and a target.
    dataset = AudioSetDataset(sample_rate=sample_rate, use_audeep=use_audeep, use_one_hot=use_one_hot, do_binary_task=do_binary_task)

    # Train sampler
    if balanced == 'none':
        Sampler = TrainSampler
    elif balanced == 'balanced':
        Sampler = BalancedTrainSampler
    elif balanced == 'alternate':
        Sampler = AlternateTrainSampler

    train_sampler = Sampler(
        indexes_hdf5_path=train_indexes_hdf5_path,
        batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size,
        black_list_csv=black_list_csv)

    # Evaluate sampler
    eval_bal_sampler = EvaluateSampler(
        indexes_hdf5_path=eval_bal_indexes_hdf5_path, batch_size=batch_size)

    # eval_test_sampler = EvaluateSampler(
    #     indexes_hdf5_path=eval_test_indexes_hdf5_path, batch_size=batch_size)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_sampler=train_sampler, collate_fn=collate_fn,
                                               num_workers=num_workers, pin_memory=True)

    eval_bal_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_sampler=eval_bal_sampler, collate_fn=collate_fn,
                                                  num_workers=num_workers, pin_memory=True)

    # eval_test_loader = torch.utils.data.DataLoader(dataset=dataset,
    #                                                batch_sampler=eval_test_sampler, collate_fn=collate_fn,
    #                                                num_workers=num_workers, pin_memory=True)

    if 'mixup' in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)

    # Evaluator
    evaluator = Evaluator(model=model, loss_type=loss_type)

    # Statistics
    statistics_container = StatisticsContainer(statistics_path)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    if use_cos_sched:
        def get_lr_lambda(nb_epochs):
            def lr_lambda(epoch):
                return (1.0 + np.cos((epoch - 1) * np.pi / nb_epochs)) * 0.5
            return lr_lambda

        lr_scheduler = LambdaLR(optimizer, get_lr_lambda(early_stop))


    if pretrain:
        if 'ResNet22Cnn10' in model_type:
            pretrained_checkpoint_path_cnn10 = '/tmpdir/pellegri/ComParE2021_PRS/dist/checkpoints/main/sample_rate=16000,window_size=1024,hop_size=320,mel_bins=64,fmin=10,fmax=8000/data_type=train/Cnn10/loss_type=clip_bce/balanced=balanced/augmentation=mixup/spec_augment=yes/signal_augmentation=none/audeep=no/batch_size=32/best_ema_model.pth'
            pretrained_checkpoint_path_resnet22 = '/tmpdir/pellegri/ComParE2021_PRS/dist/checkpoints/main/sample_rate=16000,window_size=1024,hop_size=320,mel_bins=64,fmin=10,fmax=8000/data_type=train/ResNet22/loss_type=clip_bce/balanced=balanced/augmentation=mixup/spec_augment=yes/signal_augmentation=none/audeep=no/batch_size=32/best_ema_model.pth'
            print("Loading from pretrained models: Cnn10 and ResNet22")
            logging.info('Load pretrained models from Cnn10: {}\n ResNet22: {}'.format(pretrained_checkpoint_path_cnn10, pretrained_checkpoint_path_resnet22))
            model.load_from_pretrain(pretrained_checkpoint_path_cnn10, pretrained_checkpoint_path_resnet22)

        else:
            logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
            model.load_from_pretrain(pretrained_checkpoint_path)

    # Resume training
    if resume_iteration > 0:
        resume_checkpoint_path = os.path.join(workspace, 'checkpoints', dir_path_suffix(args, pretrain=pretrain),
                                              '{}_iterations.pth'.format(resume_iteration))

        logging.info('Loading checkpoint {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        train_sampler.load_state_dict(checkpoint['sampler'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = checkpoint['iteration']

    else:
        iteration = 0

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)

    train_bgn_time = time.time()

    time1 = time.time()

    best_devel_uar = 0.
    best_ema_devel_uar = 0.

    for batch_data_dict in train_loader:
        """batch_data_dict: {
            'audio_name': (batch_size [*2 if mixup],),
            'waveform': (batch_size [*2 if mixup], clip_samples),
            'target': (batch_size [*2 if mixup], classes_num),
            (ifexist) 'mixup_lambda': (batch_size * 2,)}
        """

        # Evaluate
        if (iteration % 200 == 0 and iteration > resume_iteration) or (iteration == 0):
            train_fin_time = time.time()

            bal_statistics = evaluator.evaluate(eval_bal_loader, use_audeep=use_audeep, do_binary_task=do_binary_task)
            # test_statistics = evaluator.evaluate(eval_test_loader)

            devel_map = np.mean(bal_statistics['average_precision'])
            logging.info('Validate devel mAP: {:.3f}'.format(
                devel_map))

            logging.info('Validate devel auc: {:.3f}'.format(
                np.mean(bal_statistics['auc'])))

            devel_uar = np.mean(bal_statistics['uar'])
            logging.info('Validate devel uar: {:.3f}'.format(
                devel_uar))

            logging.info('Validate loss: {:.3f}'.format(
                bal_statistics['loss']))

            # if host_name == 'erebor':
            if not do_binary_task:
                writer.add_scalar('UAR', np.mean(bal_statistics['uar']), iteration)

            writer.add_scalar('loss_devel', bal_statistics['loss'], iteration)
            writer.add_scalar('mAP_devel', devel_map, iteration)

            # logging.info('Validate test mAP: {:.3f}'.format(
            #     np.mean(test_statistics['average_precision'])))

            statistics_container.append(iteration, bal_statistics, data_type='devel')
            # statistics_container.append(iteration, test_statistics, data_type='test')
            statistics_container.dump()

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(iteration, train_time, validate_time))

            logging.info('------------------------------------')

            train_bgn_time = time.time()

            # Save model
            if best_devel_uar < devel_uar and iteration > 10000:
                checkpoint = {
                    'iteration': iteration,
                    'model': model.module.state_dict(),
                    'sampler': train_sampler.state_dict()}

                if do_binary_task:
                    checkpoint_path = os.path.join(
                        checkpoints_dir,
                        'BINARY_{}_iterations_{:.3f}_devel_uar_{:.3f}_map.pth'.format(iteration, devel_uar, devel_map))
                else:
                    checkpoint_path = os.path.join(
                        checkpoints_dir,
                        '{}_iterations_{:.3f}_devel_uar_{:.3f}_map.pth'.format(iteration, devel_uar, devel_map))

                torch.save(checkpoint, checkpoint_path)
                logging.info('Model saved to {}'.format(checkpoint_path))

                best_devel_uar = devel_uar

        # Mixup lambda
        if 'mixup' in augmentation:
            batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                batch_size=len(batch_data_dict['waveform']))

        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

        # Forward
        model.train()

        if 'mixup' in augmentation:
            if use_audeep:
                # print("MAIN", batch_data_dict['waveform'].size(), batch_data_dict['audeep'].size(), batch_data_dict['mixup_lambda'].size())
                batch_output_dict = model(batch_data_dict['waveform'], batch_data_dict['audeep'],
                                          mixup_lambda=batch_data_dict['mixup_lambda'])
                """{'clipwise_output': (batch_size, classes_num), ...}"""
            else:
                batch_output_dict = model(batch_data_dict['waveform'],
                                          mixup_lambda=batch_data_dict['mixup_lambda'])
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': do_mixup(batch_data_dict['target'],
                                                    batch_data_dict['mixup_lambda'])}
            """{'target': (batch_size, classes_num)}"""
        else:
            if use_audeep:
                batch_output_dict = model(batch_data_dict['waveform'], batch_data_dict['audeep'], mixup_lambda=None)
                """{'clipwise_output': (batch_size, classes_num), ...}"""
            else:
                batch_output_dict = model(batch_data_dict['waveform'],  mixup_lambda=None)
                """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': batch_data_dict['target']}
            """{'target': (batch_size, classes_num)}"""

            # batch_target2_dict = {'target2': batch_data_dict['target2']}
            # """{'target2': (batch_size, classes_num)}"""

            # print("emb", batch_output_dict['embedding'].size())
        # Loss
        loss = loss_func(batch_output_dict, batch_target_dict)
        # loss2 = loss_bce_logits(batch_output_dict['binary_output'], batch_target2_dict['target2'])
        # if host_name == 'erebor':
        writer.add_scalar('train_loss', loss.item(), iteration)

        # coeff_loss1 = 0.8
        # total_loss = coeff_loss1 * loss + (1.-coeff_loss1) * loss2

        # Backward
        loss.backward()
        # print(loss.item())
        # total_loss.backward()
        # print("%.3f %.3f --> %.3f"%(loss.item(), loss2.item(), total_loss.item()))

        optimizer.step()
        optimizer.zero_grad()
        if use_cos_sched:
            lr_scheduler.step()

        # if iteration > 5000:
        if iteration > 15000:
            ema.update(model.named_parameters())

        if iteration % 100 == 0:
            print('--- Iteration: {}, train time: {:.3f} s / 100 iterations ---' \
                  .format(iteration, time.time() - time1))
            time1 = time.time()

        if iteration > 15000 and iteration % 200 == 0:
        # if iteration > 5000 and iteration % 200 == 0:
            # Validation: with EMA
            # First save original parameters before replacing with EMA version
            ema.store(model.named_parameters())
            # Copy EMA parameters to model
            ema.copy_to(model.named_parameters())

            bal_statistics = evaluator.evaluate(eval_bal_loader, use_audeep=use_audeep, do_binary_task=do_binary_task)
            # test_statistics = evaluator.evaluate(eval_test_loader)

            devel_map = np.mean(bal_statistics['average_precision'])
            logging.info('ema Validate devel mAP: {:.3f}'.format(
                devel_map))

            logging.info('ema Validate devel auc: {:.3f}'.format(
                np.mean(bal_statistics['auc'])))

            devel_uar = np.mean(bal_statistics['uar'])
            logging.info('ema Validate devel uar: {:.3f}'.format(
                devel_uar))

            logging.info('ema Validate loss: {:.3f}'.format(
                bal_statistics['loss']))

            # if host_name == 'erebor':
            writer.add_scalar('ema_UAR', np.mean(bal_statistics['uar']), iteration)
            writer.add_scalar('ema_loss_devel', bal_statistics['loss'], iteration)

            # Save model
            if best_ema_devel_uar < devel_uar:
                checkpoint = {
                    'iteration': iteration,
                    'model': model.module.state_dict(),
                    'sampler': train_sampler.state_dict()}

                checkpoint_path = os.path.join(
                    checkpoints_dir, 'ema_{}_iterations_{:.3f}_devel_uar_{:.3f}_map.pth'.format(iteration, devel_uar, devel_map))

                torch.save(checkpoint, checkpoint_path)
                logging.info('Model saved to {}'.format(checkpoint_path))

                best_ema_devel_uar = devel_uar

            # Restore original parameters to resume training later
            ema.restore(model.named_parameters())

        # Stop learning
        if iteration == early_stop:
            break

        iteration += 1


if __name__ == '__main__':
    
    print(torch.__version__)
    
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--data_type', type=str, default='train', choices=['train', 'balanced_train', 'full_train'])
    parser_train.add_argument('--sample_rate', type=int, default=32000)
    parser_train.add_argument('--window_size', type=int, default=1024)
    parser_train.add_argument('--hop_size', type=int, default=320)
    parser_train.add_argument('--mel_bins', type=int, default=64)
    parser_train.add_argument('--fmin', type=int, default=50)
    parser_train.add_argument('--fmax', type=int, default=14000)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--pretrained_checkpoint_path', type=str)
    parser_train.add_argument('--freeze_base', action='store_true', default=False)
    parser_train.add_argument('--loss_type', type=str, default='clip_bce', choices=['clip_bce', 'clip_ce', 'clip_nll'])
    parser_train.add_argument('--task', type=str, default='multi', choices=['multi', 'binary'])
    parser_train.add_argument('--balanced', type=str, default='balanced', choices=['none', 'balanced', 'alternate'])
    parser_train.add_argument('--augmentation', type=str, default='mixup', choices=['none', 'mixup'])
    parser_train.add_argument('--signal_augmentation', type=str, default='none', choices=['none', 'time_stretch', 'pitch_shift', 'both'])
    parser_train.add_argument('--spec_augment', type=str, default='yes', choices=['no', 'yes'])
    parser_train.add_argument('--audeep', type=str, default='no', choices=['no', 'yes'])
    parser_train.add_argument('--batch_size', type=int, default=32)
    parser_train.add_argument('--learning_rate', type=float, default=1e-3)
    parser_train.add_argument('--use_cos_sched', type=str, default='no', choices=['no', 'yes'])
    parser_train.add_argument('--resume_iteration', type=int, default=0)
    parser_train.add_argument('--early_stop', type=int, default=1000000)
    parser_train.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')