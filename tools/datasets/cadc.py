import os
from glob import glob
import numpy as np
from .base import Base


WIDTH = 2048
INC = np.deg2rad([
    -25.010, -15.639, -11.311, -8.843, -7.255, -6.148, -5.334, -4.667,
    -4.000, -3.667, -3.334, -3.000, -2.667, -2.333, -2.001, -1.667,
    -1.333, -1.000, -0.667, -0.333, -0.000, 0.332, 0.667, 1.000,
    1.332, 1.667, 2.333, 3.333, 4.667, 7.000, 10.334, 15.000
])

class CADC(Base):
    def __init__(self, data_dir, name='CADC', inc=INC, width=WIDTH, training=True, skip=1, return_points=False, filter=''):
        super().__init__(data_dir, name=name, inc=inc, width=width, training=training, skip=skip, return_points=return_points, filter=filter)

    @staticmethod
    def get_file_id(file_name):
        # date, drive_id, drive_type, frame_id
        return os.path.join(*[file_name.split('/')[k] for k in [-6, -5, -4, -1]])

    def read_file_list(self, data_dir, train_ratio=0.8):
        fn_annotations = []
        for date in ['2018_03_06', '2018_03_07', '2019_02_27']:
            fn_ann = sorted(glob(os.path.join(data_dir, date, '*/3d_ann.json')))
            n_train = np.round(len(fn_ann) * train_ratio).astype(np.uint32)
            if self.training:
                fn_annotations += fn_ann[:n_train]
            else:
                fn_annotations += fn_ann[n_train:]

        fn_points = []
        for f in fn_annotations:
            p = 'raw/lidar_points_corrected/data/*.bin'

            fn_points += sorted(glob(f.replace('3d_ann.json', p)))

        return fn_points
