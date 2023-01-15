dataset=./experiments/human_nerf/zju_mocap/p377/single_gpu/latest4/movement
target=./CT-Net/dataset
name=/zju_mocap/p377

cd ./pytorch-openpose
conda activate pytorch-openpose
python demo.py --path $dataset --save $target/pose$name --img $target/img$name
conda deactivate

cd ./LIP_JPPNet
conda activate jppnet
python -W ignore evaluate_parsing_JPPNet-s2.py --path $dataset --txt $target/img$name/dataset.txt --out $target/seg_and_dp$name
conda deactivate


cd ./DensePose
conda activate vibe-env
python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml \
https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \
$dataset --output $target/seg_and_dp$name/dump.txt -v
conda deactivate

cd ..