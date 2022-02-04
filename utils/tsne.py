import numpy as np
from sklearn.manifold import TSNE

import torch

import matplotlib.pyplot as plt
import matplotlib

embdir='/tmpdir/pellegri/ComParE2021_PRS/dist/checkpoints/main/sample_rate=16000,window_size=1024,hop_size=320,mel_bins=64,fmin=10,fmax=8000/data_type=train/ResNet22/loss_type=clip_bce/balanced=balanced/augmentation=mixup/spec_augment=yes/signal_augmentation=none/audeep=no/batch_size=32'
emb = np.load(embdir + "/emb_devel.npy")

print(emb.shape)

targets = torch.load(embdir + "/devel_target.pth")
targets = np.squeeze(targets)
print(targets.shape)

print(targets[:10])

# emb2d = TSNE(n_components=2).fit_transform(emb[:1000])
emb2d = TSNE(n_components=2).fit_transform(emb)
print("fit _transform DONE")

classes = ["B", "C", "G", "M", "R"]
colors = ['red','green','blue','purple', 'black']

# plt.scatter(emb2d[:,0], emb2d[:,1], c=targets[:1000], cmap=matplotlib.colors.ListedColormap(colors))
# plt.scatter(emb2d[:,0], emb2d[:,1], c=targets, cmap=matplotlib.colors.ListedColormap(colors))


fig, ax = plt.subplots()
for i in range(len(colors)):
    ind = targets == i
    ax.scatter(emb2d[ind,0], emb2d[ind,1], c=colors[i], label=classes[i],
               alpha=0.3, edgecolors='none')
ax.legend()
# ax.grid(True)

plt.savefig("results/emb.png")
# plt.show()
