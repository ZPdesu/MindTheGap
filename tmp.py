

import numpy as np

import torch

from glob import glob
import os

ckpt_path = os.path.join('pretrained_models', 'ffhq.pt')
ckpt = torch.load(ckpt_path)
latent_avg = ckpt['latent_avg']

npy_list = glob(os.path.join('tmp', '*.npy'))

for i in npy_list:
    name = os.path.basename(i)


    ckpt_path = os.path.join('pretrained_models', name[:-4] + '.pt')
    ckpt = torch.load(ckpt_path)
    ckpt['latent_avg'] = latent_avg.cuda()

    torch.save(ckpt, ckpt_path)



print(100)