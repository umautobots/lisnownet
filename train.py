#!/usr/bin/env python3
import argparse
import time
import os
from glob import glob
import json
from collections import defaultdict
from datetime import datetime
from pprint import pprint
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tools import losses
from tools.datasets.wads import WADS
from tools.datasets.cadc import CADC
from tools.models import MWCNN

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=20,
                    help='Number of epochs. (default: 20)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size in each training step. (default: 32)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate. (default: 1e-3)')
parser.add_argument('--lr_decay', type=float, default=0.95,
                    help='Learning rate decay.')
parser.add_argument('--alpha', type=float, default=5.0)
parser.add_argument('--beta', type=float, default=0.5,
                    help='Relative weight of the FFT loss. Must be between 0 and 1.')
parser.add_argument('--dataset', type=str, default='cadc', choices=['cadc', 'wads'])
parser.add_argument('--log_dir', type=str, default='./logs')
parser.add_argument('--tag', type=str, default='')
config = vars(parser.parse_args())

device = torch.device('cuda')

config['tag'] = config['tag'].split('/')[-1]
if not config['tag'].strip():
    config['tag'] = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

log_dir = os.path.join(config['log_dir'], config['tag'])
os.makedirs(log_dir, exist_ok=True)

assert config['alpha'] > 1.0
assert 0 <= config['beta'] <= 1

w1 = (2**config['alpha'] - 1) / 2**config['alpha']

config_file = os.path.join(log_dir, 'config.json')
if os.path.exists(config_file):
    # Overwrite with saved config
    with open(config_file, 'r') as f:
        config = json.load(f)
else:
    # Save config to a JSON file
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

pprint(config)

if config['lr_decay'] < 0:
    config['lr_decay'] = 0.1**(1 / config['num_epochs'])

if config['dataset'] == 'cadc':
    ds_train = CADC('./data/cadcd', training=True)
    ds_val = CADC('./data/cadcd', training=False)
elif config['dataset'] == 'wads':
    ds_train = WADS('./data/wads', training=True)
    ds_val = WADS('./data/wads', training=False)

loader_train = DataLoader(
    ds_train,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    drop_last=True
)

loader_val = DataLoader(
    ds_val,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    drop_last=False
)

# Using multiple GPUs
model = nn.DataParallel(
    MWCNN(),
    device_ids=range(torch.cuda.device_count())
).to(device)

checkpoints = sorted(glob(os.path.join(log_dir, '*.pth')))
if len(checkpoints):
    ckpt = checkpoints[-1]
    print(f'Loading the last checkpoint {ckpt:s}')
    model.load_state_dict(torch.load(ckpt))

    start_epoch = int(os.path.basename(ckpt).split('.')[0])
else:
    start_epoch = 0

writer = SummaryWriter(log_dir)

lr0 = config['lr'] * config['lr_decay']**start_epoch
optimizer = torch.optim.Adam(model.parameters(), lr=lr0)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config['lr_decay'])

t_start = time.time()
for epoch in range(start_epoch, config['num_epochs']):
    tb_train = defaultdict(list)
    model.train()
    for index, (fid, x, _, _) in enumerate(loader_train):
        x = x.to(device)

        # Forward
        idx_valid, y = model(x)
        l_dwt, l_fft = losses.sparsity_loss(y)

        residual = (y - x) * idx_valid
        l_res = residual.abs().sum() / idx_valid.sum()

        l_sp = config['beta'] * l_dwt + (1 - config['beta']) * l_fft
        loss = w1 * l_sp + (1 - w1) * l_res

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # elapsed time
        mm, ss = divmod(time.time() - t_start, 60)
        hh, mm = divmod(mm, 60)

        print(' '.join([
            f"[{(epoch + 1):4d}/{config['num_epochs']:4d}]",
            f'[{(index + 1):4d}/{len(loader_train):4d}]',
            f'[{int(hh):02d}h{int(mm):02d}m{int(ss):02d}s]',
            f'losses: {l_dwt.item():.6f} {l_fft.item():.6f} {l_res.item():.6f}'
        ]), end='\t\r')

        tb_train['loss/train/DWT'].append(l_dwt.item())
        tb_train['loss/train/FFT'].append(l_fft.item())
        tb_train['loss/train/Residual'].append(l_res.item())

    for key, value in tb_train.items():
        writer.add_scalar(key, np.nanmean(value), epoch + 1)

    fn_ckpt = os.path.join(log_dir, f'{(epoch + 1):04d}.pth')
    print(f'\nSaving {fn_ckpt:s} ...')
    torch.save(model.state_dict(), fn_ckpt)

    tb_val = defaultdict(list)
    model.eval()
    for index, (fid, x, _, _) in enumerate(loader_val):
        with torch.no_grad():
            x = x.to(device)

            idx_valid, y = model.module.forward(x)
            l_dwt, l_fft = losses.sparsity_loss(y)

            residual = (y - x) * idx_valid
            l_res = residual.abs().sum() / idx_valid.sum()

            tb_val['loss/val/DWT'].append(l_dwt.item())
            tb_val['loss/val/FFT'].append(l_fft.item())
            tb_val['loss/val/Residual'].append(l_res.item())

    for key, value in tb_val.items():
        writer.add_scalar(key, np.nanmean(value), epoch + 1)

    writer.flush()
    scheduler.step()

writer.close()
