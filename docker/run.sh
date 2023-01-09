#!/bin/bash

PASSWD_FILE=$(mktemp)
GROUP_FILE=$(mktemp)
echo $(getent passwd $(id -un)) > $PASSWD_FILE
echo $(getent group $(id -un)) > $GROUP_FILE

[ -z $DATA_PATH ] && DATA_PATH=$(mktemp -d)
[ -z $1 ] && GPU="all" || GPU=$1

xhost +local:

docker run -it --rm \
    --gpus '"device='$GPU'"' \
    --name lisnownet-${GPU//,/}-$(openssl rand -hex 4) \
    --hostname $(hostname) \
    --ipc host \
    -e DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e HOME \
    -e XDG_RUNTIME_DIR=/run/user/$(id -u) \
    -u $(id -u):$(id -g) \
    -v /run/user/$(id -u):/run/user/$(id -u) \
    -v /etc/timezone:/etc/timezone:ro \
    -v /etc/localtime:/etc/localtime:ro \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $DATA_PATH:$DATA_PATH:ro \
    -v $PASSWD_FILE:/etc/passwd:ro \
    -v $GROUP_FILE:/etc/group:ro \
    -v $(pwd)/docker/home:$HOME \
    -v $(pwd):/workspace \
    -w /workspace \
    lisnownet:latest

xhost -local:
