#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/refsegrs.yaml
save_path=exps/RefSegRS/LSCF-$now

mkdir -p $save_path

python -m torch.distributed.run \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    train.py \
    --config=$config \
    --save-path $save_path --port $2 \
    2>&1 | tee $save_path/$now.log
