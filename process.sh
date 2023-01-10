#!/bin/bash

# 自定义视频id
data_id=jntm

# 给定数据集参数
dataset_path=../video
data_name=$data_id.mp4
target_folder=../dataset/wild/$data_id
mkdir -p $target_folder

# SMPL参数估计
conda activate vibe
cd VIBE
python demo.py --vid_file $dataset_path/$data_name --output_folder output/ --image_folder $target_folder/images --sideview
python process.py --data_id $data_id --output_path $target_folder/
conda deactivate

# Mask输出
conda activate vibe-env
cd ../detectron2/demo
mkdir -p ../$target_folder/masks
python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --input ../$target_folder/images/*.png --output ../$target_folder/masks  --opts MODEL.WEIGHTS ./model_final_2d9806.pkl
conda deactivate
cd ../..

conda activate humannerf
cd ./tools/prepare_wild
python prepare_dataset.py --subject $data_id
cd ../..