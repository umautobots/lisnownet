import os
from glob import glob
import numpy as np
from .base import Base


WIDTH = 2048
INC = np.deg2rad([
    -24.937, -18.929, -13.970, -13.014, -12.046, -11.072, -10.085, -9.100,
    -8.099, -7.103, -6.101, -5.938, -5.766, -5.605, -5.431, -5.269,
    -5.097, -4.932, -4.760, -4.598, -4.425, -4.261, -4.090, -3.924,
    -3.752, -3.588, -3.415, -3.250, -3.080, -2.913, -2.740, -2.576,
    -2.405, -2.238, -2.068, -1.900, -1.728, -1.562, -1.391, -1.224,
    -1.053, -0.885, -0.715, -0.548, -0.377, -0.209, -0.040, 0.129,
    0.297, 0.468, 0.635, 0.806, 0.973, 1.144, 1.311, 1.482,
    1.648, 1.820, 1.988, 3.000, 5.017, 8.019, 10.992, 14.842
])


class WADS(Base):
    def __init__(self, data_dir, name='WADS', inc=INC, width=WIDTH, training=True, skip=1, return_points=False, filter=''):
        super().__init__(data_dir, name=name, inc=inc, width=width, training=training, skip=skip, return_points=return_points, filter=filter)

    @staticmethod
    def read_files(file_name):
        points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)

        # another dirty fix for WADS :(
        # at the time of publishing this work, there are duplicated points in every file.
        points, idx_unique = np.unique(points, axis=0, return_index=True)
        points[:, -1] /= 255

        file_name = file_name.replace('velodyne', 'labels').replace('.bin', '.label')
        labels = np.fromfile(file_name, dtype=np.int32)[idx_unique]

        return points, labels

    def read_file_list(self, data_dir):
        if self.training:
            sequences = [11, 12, 13, 14]
            sequences += [16, 17, 18, 20]
            sequences += [23, 24, 26, 28]
            sequences += [34, 35, 36, 76]
        else:
            sequences = [15, 22, 30]
            # 37 seems to be an exact copy of another sequence ...
            # sequences += [37]

        fn_points = []
        for seq in sequences:
            p = os.path.join(data_dir, f'{seq:02d}/velodyne/*.bin')
            fn_points += sorted(glob(p))

        return fn_points
