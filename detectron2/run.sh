cd projects/DensePose
python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml \
https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \
/home/wenhao/CV/try/image --output /home/wenhao/CV/try/detect/dump.txt -v