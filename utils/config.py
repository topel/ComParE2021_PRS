
import numpy as np
import csv

sample_rate = 16000

# clip_samples = sample_rate * 10
# Audio clips are of variable-length, but max length is 47860 samples, 3s long
# /homelocal/thomas/data/ComParE2021_PRS/dist/wav/devel_00756.wav
clip_samples = 47860
audeep_dim = 4096

# Load label
with open('metadata/class_labels_indices.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    lines = list(reader)

labels = []
ids = []    # Each label has a unique id such as "/m/068hy"
for i1 in range(1, len(lines)):
    id = lines[i1][1]
    label = lines[i1][2]
    ids.append(id)
    labels.append(label)

classes_num = len(labels)

lb_to_ix = {label : i for i, label in enumerate(labels)}
ix_to_lb = {i : label for i, label in enumerate(labels)}

id_to_ix = {id : i for i, id in enumerate(ids)}
ix_to_id = {i : id for i, id in enumerate(ids)}

full_samples_per_class = np.array([
    3458, 2217, 158, 874, 208
])