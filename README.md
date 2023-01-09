# LiSnowNet: Real-time Snow Removal for LiDAR Point Cloud

This is the offical implementation of [LiSnowNet: Real-time Snow Removal for LiDAR Point Clouds](https://ieeexplore.ieee.org/document/9982248).

- Results for [CADC](http://cadcd.uwaterloo.ca/) ![cadc_single_frame](figures/cadc-2019_02_27-0082.gif)

- Results for [WADS](https://digitalcommons.mtu.edu/wads/) ![wads_single_frame](figures/wads-15.gif)

## Requirements

- Ubuntu 18.04+
- NVIDIA driver >= 515
- Docker with the [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/)
- [NVIDIA Container Runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

## Installation

- Build the docker image

    ```bash
    $ docker build --tag lisnownet -f docker/Dockerfile .
    ```

- Launch a container

    ```bash
    $ DATA_PATH=/path/to/datasets     # the dataset path to be mounted to the container
    $ ./docker/run.sh       # use all GPUs
    $ ./docker/run.sh 0     # use GPU #0
    $ ./docker/run.sh 2,3   # use GPU #2 and #3
    ```

## Datasets

Download the [Canadian Adverse Driving Conditions (CADC) Dataset](http://cadcd.uwaterloo.ca/) and the [Winter Adverse Driving dataSet (WADS)](https://digitalcommons.mtu.edu/wads/), and create symlinks to them under the `data` folder:

```
./data
├── cadcd
|   └── {DATE}/{DRIVE_ID}/raw/lidar_points/corrected/data/{FRAME_ID}.bin
└── wads
    └── {DRIVE_ID}
        ├── labels/{FRAME_ID}.label
        └── velodyne/{FRAME_ID}.bin
```

## Train

To train the model, run

```bash
$ ./train.py [--batch_size BATCH_SIZE] [--dataset DATASET] [--alpha ALPHA] [--tag TAG] [...]
```

For example:

```bash
$ ./train.py --dataset cadc --tag cadc_alpha=5.0 --lr_decay -1 --alpha 5.0
```

## Evaluate

```bash
$ ./eval.py [--batch_size BATCH_SIZE] [--dataset DATASET] [--tag TAG] [--threshold THRESHOLD] [...]
```

For example:

```bash
$ ./eval.py --tag wads_alpha=4.0 --batch_size 8 --dataset wads --threshold 8e-3
$ ./eval.py --tag cadc_alpha=5.0 --batch_size 8 --dataset cadc --threshold 1.2e-2
```

## Citation

```
@INPROCEEDINGS{9982248,
    author={Yu, Ming-Yuan and Vasudevan, Ram and Johnson-Roberson, Matthew},
    booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    title={LiSnowNet: Real-time Snow Removal for LiDAR Point Clouds},
    year={2022},
    volume={},
    number={},
    pages={6820-6826},
    doi={10.1109/IROS47612.2022.9982248}}
```
