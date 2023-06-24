#!/usr/bin/env bash

#set -x
#
#EXP_DIR=exps/r50_deformable_detr
#PY_ARGS=${@:1}
#
#python -u main.py \
#    --output_dir ${EXP_DIR} \
#    ${PY_ARGS}

# local
python ../main.py \
    --coco_path /root/autodl-tmp/data/coco/ \
    --resum /root/autodl-tmp/model/deformable-detr/r50_deformable_detr-checkpoint.pth
    --device cpu \
    --output_dir ./output

# service
--coco_path /root/autodl-tmp/data/coco/
--resum /root/autodl-tmp/model/deformable-detr/r50_deformable_detr-checkpoint.pth
--device cuda
--epochs 100
--output_dir ../output

# local
--coco_path /Users/ningyuguang/data/OD/minicoco/
--num_workers 1
--batch_size 1
--device cpu
--epochs 100
--output_dir ../output

--resum /Users/ningyuguang/Model/pretrain/OD/r50_deformable_detr-checkpoint.pth
