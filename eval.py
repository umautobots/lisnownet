#!/usr/bin/env python3
import argparse
import time
from multiprocessing import Pool, cpu_count
import os
from glob import glob
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tools.datasets.wads import WADS
from tools.datasets.cadc import CADC
from tools.models import MWCNN
from tools.utils import image2points


def save_results(frame, log_dir, zmin=-2, zmax=6, axlim=30):
    # points: [x, y, z, i, delta_d, delta_i, gt, pr]
    fid, points = frame
    points = points[::-1, :]
    print(f'\t\t\t{fid}', end='\r')

    xyzi, res = points[:, :4], points[:, 4:6]
    idx_pr = points[:, 7].astype(bool)

    # save filtered points
    fname_xyz = os.path.join(
        log_dir,
        os.path.dirname(fid),
        'velodyne',
        os.path.basename(fid)
    )
    os.makedirs(os.path.dirname(fname_xyz), exist_ok=True)
    xyzi[~idx_pr, :].tofile(fname_xyz)

    # save BEV
    fname_png = fname_xyz.replace('velodyne', 'bev').replace('.bin', '.png')
    os.makedirs(os.path.dirname(fname_png), exist_ok=True)

    fid_list = fid.replace('.bin', '').split('/')
    if len(fid_list) == 2:
        # WADS
        drive_id, frame_id = fid_list
    elif len(fid_list) == 4:
        # CADC
        _, drive_id, _, frame_id = fid_list
    else:
        raise ValueError

    drive_id, frame_id = int(drive_id), int(frame_id)

    figure_id = drive_id << 4 + frame_id
    fig = plt.figure(figure_id, figsize=(8, 4.5), tight_layout=True)
    axes = [fig.add_subplot(1, 2, k + 1) for k in range(2)]

    for idx, ax in enumerate(axes):
        if idx:
            ax.set_title('Denoised')
            ax.set_yticklabels([])
            points = xyzi[~idx_pr, :]
        else:
            ax.set_title('Raw')
            ax.set_ylabel(r'$y$ [m]')
            points = xyzi

        ax.scatter(
            points[:, 0], points[:, 1], c=points[:, 2],
            s=0.1, vmin=zmin, vmax=zmax, alpha=0.9, marker=','
        )

        ax.axis('scaled')
        ax.set_xlim(-axlim, axlim)
        ax.set_ylim(-axlim, axlim)
        ax.set_xlabel(r'$x$ [m]')

    fig.savefig(fname_png, dpi=240)
    plt.close(fig)


def benchmark(frame):
    # frame[0] is frame_id, frame[1] is the points with GT and PR
    points = frame[1]
    idx_gt, idx_pr = points[:, 6].astype(bool), points[:, 7].astype(bool)

    tp = (idx_pr & idx_gt).sum()
    fp = (idx_pr & ~idx_gt).sum()
    fn = (~idx_pr & idx_gt).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    iou = tp / (idx_pr | idx_gt).sum()

    return precision, recall, iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--threshold', type=float, default=1.2e-2)
    parser.add_argument('--z_ground', type=float, default=-1.8)
    parser.add_argument('--snow_id', type=int, default=110)
    parser.add_argument('--dataset', type=str, default='cadc', choices=['cadc', 'wads'])
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--tag', type=str, default='')
    config = parser.parse_args()

    config.tag = config.tag.split('/')[-1]
    log_dir = os.path.join(config.log_dir, config.tag)

    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica'],
        'font.size': 10
    })

    device = torch.device('cuda')

    d_thresh, i_thresh = 2.5, 2 / 255

    if config.dataset == 'cadc':
        dataset = CADC(data_dir='./data/cadcd', training=False, skip=1)
    elif config.dataset == 'wads':
        dataset = WADS(data_dir='./data/wads', training=False, skip=1)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=cpu_count() // 2,
        pin_memory=False,
        drop_last=False
    )

    # Using multiple GPUs
    model = nn.DataParallel(
        MWCNN(),
        device_ids=range(torch.cuda.device_count())
    ).to(device)

    checkpoints = sorted(glob(os.path.join(log_dir, '*.pth')))
    ckpt = checkpoints[-1]

    if len(checkpoints):
        print(f'\nLoading the last checkpoint {ckpt:s}')
        model.load_state_dict(torch.load(ckpt))
    else:
        raise FileExistsError(f'{ckpt:s} doe not exist.')

    model.eval()

    runtime, frames = [], []
    for index, (fid, range_img, xyz_img, lbl_img) in enumerate(loader):
        range_img = range_img.to(device)
        xyz_img, lbl_img = xyz_img.to(device), lbl_img.to(device)

        # Forward
        t0 = time.time()

        idx_valid, y = model(range_img)
        residual_img = (y - range_img) * idx_valid
        delta_d, delta_i = [residual_img[:, k, :, :] for k in range(2)]

        # convert back to actual readings
        range_img = range_img.pow(3)

        # predictions
        pr_img = delta_d * delta_i.pow(3) > config.threshold
        # snowflakes are higher than the ground plane
        pr_img &= xyz_img[:, 2, :, :] > config.z_ground
        # snowflakes are very dark
        pr_img &= range_img[:, 1, :, :] < i_thresh
        # points within a small distance are 100% snowflakes
        pr_img |= range_img[:, 0, :, :] < d_thresh

        runtime.append((time.time() - t0) / range_img.shape[0])

        # results to be saved
        gt_img, pr_img = (lbl_img == config.snow_id), pr_img.unsqueeze(1)
        output_img = torch.cat([
            xyz_img,
            range_img[:, 0, :, :].unsqueeze(1),
            residual_img,
            gt_img,
            pr_img
        ], dim=1)
        idx_valid = idx_valid[:, 0, :, :].unsqueeze(1).expand_as(output_img)
        output_img[~idx_valid] = -1

        p_out = image2points(output_img)
        p_out = p_out.detach().cpu().numpy()
        for _fid, p1 in zip(fid, p_out):
            p1 = p1[np.isfinite(p1).all(axis=-1), :]

            print(', '.join([
                f'[{index + 1:4d}/{len(loader):4d}] {_fid}',
                f'FPS = {1 / np.median(runtime):.4f}',
                f'num_points = {p1.shape[0]:d}'
            ]), end='\r')

            frames.append((_fid, p1))

    print('')
    num_proc = min(3 * cpu_count() // 4, 64)

    if config.dataset == 'wads':
        with Pool(num_proc) as pool:
            out = pool.map(benchmark, frames)

        precision = np.array([o[0] for o in out])
        recall = np.array([o[1] for o in out])
        iou = np.array([o[2] for o in out])
        score = precision * recall * iou

        print('\n'.join([
            f'>>> Precision:\t{np.mean(precision):.4f} +/- {np.std(precision):.4f}',
            f'>>> Recall:\t{np.mean(recall):.4f} +/- {np.std(recall):.4f}',
            f'>>> IOU:\t{np.mean(iou):.4f} +/- {np.std(iou):.4f}',
            f'>>> Score:\t{np.mean(score):.4f} +/- {np.std(score):.4f}',
            f'>>> Runtime:\t{1000 * np.median(runtime):.4f} ms'
        ]))
    else:
        print(f'No GT point-wise labels for {dataset.name:s}. Skipping the de-noising benchmark.')

    print('Saving results ... ', end='\r')
    with Pool(num_proc) as pool:
        pool.map(partial(save_results, log_dir=log_dir), frames)

    print('\nDone.')
