#!/usr/bin/env bash

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/cpfs01/user/wangbangjun/opt/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/cpfs01/user/wangbangjun/opt/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/cpfs01/user/wangbangjun/opt/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/cpfs01/user/wangbangjun/opt/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate roadperception

timestamp=`date +"%y%m%d.%H%M%S"`
WORK_DIR=$(dirname $(readlink -f "$0"))
CODE_HOME=/cpfs01/user/wangbangjun/StreamMapNet/
CONFIG=plugin/configs/av2_unimapping_608_100x50_24e.py

export PYTHONPATH=$CODE_HOME:$PYTHONPATH

export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_NET_PLUGIN=none

#CONFIG=$1
#GPUS=$2
GPUS=8
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
