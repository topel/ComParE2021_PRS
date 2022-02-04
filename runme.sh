#!/bin/bash

# change these paths to point to your ComParE2021_PRS/dist directory:
# on erebor:
DATASET_DIR="/homelocal/thomas/data/ComParE2021_PRS/dist"
WORKSPACE="/homelocal/thomas/data/ComParE2021_PRS/dist"

# ============ Pack waveform and target to hdf5 ============

# Pack evaluation waveforms to a single hdf5 file: run the following commented lines for the three subsets train, devel and test:
#subset=train
subset=devel
#subset=test

#python3 utils/dataset.py pack_waveforms_to_hdf5 --csv_path=$DATASET_DIR"/lab/$subset.csv" --audios_dir=$DATASET_DIR"/wav" \
#     --waveforms_hdf5_path=$DATASET_DIR/"hdf5s/waveforms/$subset.h5"

# optional: create a "mini-data" subset
#python3 utils/dataset.py pack_waveforms_to_hdf5  --mini_data --csv_path=$DATASET_DIR"/lab/$subset.csv" --audios_dir=$DATASET_DIR"/wav" --waveforms_hdf5_path="hdf5s/waveforms/$subset.h5"

# ============ Prepare training indexes ============
#python3 utils/create_indexes.py create_indexes --waveforms_hdf5_path=$DATASET_DIR"/hdf5s/waveforms/$subset.h5" --indexes_hdf5_path=$DATASET_DIR"/hdf5s/indexes/$subset.h5"

# optional: create a "mini-data" subset
#python3 utils/create_indexes.py create_indexes --waveforms_hdf5_path="hdf5s/waveforms/$subset.h5.mini" --indexes_hdf5_path="hdf5s/indexes/$subset.h5.mini"

### ============ Train & Inference ============

# without mixup:
#augmentation="none"
#loss_type="clip_ce"

# with mixup:
augmentation='mixup'
loss_type='clip_bce'
use_cos_sched='no'

python3 pytorch/main.py train \
    --workspace=$WORKSPACE \
    --data_type='train' \
    --sample_rate=16000\
    --window_size=1024 \
    --hop_size=320 \
    --mel_bins=64 \
    --fmin=10 \
    --fmax=8000 \
    --model_type='Cnn10'\
    --task='multi' \
    --loss_type=$loss_type \
    --balanced='balanced' \
    --audeep='no' \
    --augmentation=$augmentation \
    --signal_augmentation='none' \
    --spec_augment='yes'\
    --batch_size=32 \
    --resume_iteration=0 \
    --learning_rate=1e-3 \
    --use_cos_sched=$use_cos_sched \
    --early_stop=60000 \
    --cuda

#    --pretrained_checkpoint_path='yes' \
#    --model_type='Transfer_ResNet22'\
#    --pretrained_checkpoint_path=$PRETRAINED_DIR'/ResNet22_mAP=0.430.pth' \
#    --freeze_base \

# ============  Plot statistics  ============
#python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select=1_mobilenet

## ============ Inference with trained model ============
### Inference audio tagging with trained model

# select a subset on which you want to make predictions
subset=devel
#subset=test

#python3 pytorch/main_inference.py inference \
#    --workspace=$WORKSPACE \
#    --data_type=$subset \
#    --sample_rate=16000\
#    --window_size=1024 \
#    --hop_size=320 \
#    --mel_bins=64 \
#    --fmin=10 \
#    --fmax=8000 \
#    --model_type='Cnn10' \
#    --checkpoint_name='best_ema_model_936.pth' \
#    --loss_type='clip_bce' \
#    --balanced='balanced' \
#    --audeep='no'\
#    --augmentation='mixup' \
#    --signal_augmentation='none' \
#    --use_cos_sched='no' \
#    --spec_augment='yes'\
#    --batch_size=32 \
#    --save_embeddings_to_disk='no' \
#    --cuda

