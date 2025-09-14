#!/bin/bash

vox1_dev_path=/home/fangzh21/data/voxceleb1/dev/wav
vox2_dev_path=/home/fangzh21/data/voxceleb2/dev/aac
vox_test_path=/mnt/database/sre/voxceleb/1/test/wav
vox_test_list=/home/fangzh21/data/vox_file/veri_test2.txt

dataset=vox2  # vox1, vox2, cn
batch_size=256
gpu=5
start=2
end=2

# 准备训练数据
if [ $dataset == "vox1" ]; then
    root=$vox1_dev_path
elif [ $dataset == "vox2" ]; then
    root=$vox2_dev_path
else
    echo "dataset error!"
    exit
fi

if [ $start -le 1 ] && [ 1 -le $end ]; then
    echo "build train list..."
    mkdir -p data
    python steps/build_vox_train_list.py \
        --root $root \
        --save_path "data/$dataset.txt"
fi

if [ $start -le 2 ] && [ 2 -le $end ]; then
    echo "train model..."
    echo "CUDA: $gpu, dataset: $dataset"
    # 执行训练脚本
    CUDA_VISIBLE_DEVICES=$gpu python train_standard.py \
                                --dataset $dataset \
                                --batch_size $batch_size \
                                --train_list "data/$dataset.txt" \
                                --test_list $vox_test_list

    echo "training finish!"
fi


