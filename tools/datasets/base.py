import os
from glob import glob
import numpy as np
from scipy.interpolate import interp1d
from torch.utils.data import Dataset


WIDTH = 2048
INC = np.deg2rad(np.linspace(-24, 3, 64))


class Base(Dataset):
    def __init__(self, data_dir, name='Base', inc=INC, width=WIDTH, training=True, skip=1, return_points=False, filter=''):
        super().__init__()

        self.name = name
        self.width = width
        self.inc = inc
        self.num_beams = self.inc.size

        self.inc2ring = interp1d(
            self.inc,
            np.arange(self.num_beams),
            kind='quadratic',
            fill_value='extrapolate'
        )

        self.training = training
        self.return_points = return_points

        self.fn_points = self.read_file_list(data_dir)[::skip]
        self.fn_points = [fn for fn in self.fn_points if filter in fn]

        assert len(self.fn_points) > 0

        self.rng = np.random.default_rng()
        self.shrink = np.cbrt

    def __len__(self):
        return len(self.fn_points)

    def __getitem__(self, index, min_keep_rate=0.9):
        fname = self.fn_points[index]
        fid = self.get_file_id(fname)
        points, labels = self.read_files(fname)

        # a dirty fix for WADS :(
        # at the time of publishing this work, there are many accumulated snow points being
        # labeled as active (i.e. in the air, id = 110)
        idx_mislabeled = (points[:, 3] > 1 / 255) & (labels == 110)
        labels[idx_mislabeled] = 255

        if self.training:
            # random drop
            keep_rate = min_keep_rate + (1 - min_keep_rate) * self.rng.random()
            idx_keep = (self.rng.random(points.shape[0]) < keep_rate)
            points, labels = points[idx_keep, :], labels[idx_keep]

            # random flip
            points[:, 0] *= np.sign(self.rng.random() - 0.5)
            points[:, 1] *= np.sign(self.rng.random() - 0.5)

        if self.return_points:
            return fid, points, labels
        else:
            return [fid] + list(self.points2image(points, labels))

    def points2image(self, points, labels, interleave=True):
        '''
        Input
            - points:       (num_points, 4)         np.float32
            - labels:       (num_points,)           np.int32

        Output
            - range_img:    (2, num_beams, width)   points.dtype
            - xyz_img:      (3, num_beams, width)   points.dtype
            - lbl_img:      (1, num_beams, width)   labels.dtype
        '''

        depth = np.linalg.norm(points[:, :3], axis=-1)
        if self.training:
            order = np.arange(depth.size, dtype=np.int32)
            self.rng.shuffle(order)
        else:
            order = np.argsort(depth)[::-1]
            if interleave:
                num_split = self.num_beams * 4
                order = np.hstack([order[k::num_split] for k in range(num_split)])

        points, labels = points[order, :], labels[order]
        depth = depth[order]

        inclination = np.arcsin(points[:, 2] / depth)
        azimuth = np.arctan2(points[:, 1], points[:, 0])

        ring = self.inc2ring(inclination).round().astype(np.int32)
        i0 = (self.num_beams - 1) - ring
        i1 = 1 - 0.5 * (azimuth / np.pi + 1)
        i1 = np.floor(i1 * self.width).astype(np.int32)

        idx_valid = (i0 >= 0) & (i0 < self.num_beams)
        idx_valid &= (i1 >= 0) & (i1 < self.width)
        i0, i1 = i0[idx_valid], i1[idx_valid]

        range_img = np.full([2, self.num_beams, self.width], -1, dtype=points.dtype)
        range_img[0, i0, i1] = self.shrink(depth[idx_valid])
        range_img[1, i0, i1] = self.shrink(points[idx_valid, -1])

        xyz_img = np.full([3, self.num_beams, self.width], -np.inf, dtype=points.dtype)
        for c in range(3):
            xyz_img[c, i0, i1] = points[idx_valid, c]

        lbl_img = np.full([self.num_beams, self.width], -1, dtype=labels.dtype)
        lbl_img[i0, i1] = labels[idx_valid]
        lbl_img = np.expand_dims(lbl_img, 0)

        return range_img, xyz_img, lbl_img

    @staticmethod
    def get_file_id(file_name):
        return os.path.join(*[file_name.split('/')[k] for k in [-3, -1]])

    @staticmethod
    def read_files(file_name):
        points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
        labels = np.full(points.shape[0], -1, dtype=np.int32)

        return points, labels

    def read_file_list(self, data_dir):
        if self.training:
            p = os.path.join(data_dir, 'training/velodyne/*.bin')
        else:
            p = os.path.join(data_dir, 'testing/velodyne/*.bin')

        return sorted(glob(p))
