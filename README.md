# ClothesmanNERF

这是我们CV作业的github储存仓库，用于实现从两个视频中提取人物进行三维建模并进行姿势与服装的迁移。

## Prerequisite

### Configure environment

请确认您已经配置好我们所要求的环境：
如要成功运行本代码，需准备好六个conda环境：vibe, detectron2, openpose, jppnet, humannerf, ctnet。 前四个环境将用来数据集生成， 后两个环境用来运行我们的代码。关于前四个环境的搭建，我们强烈建议您git clone前两个开源仓库的代码并安装所需环境。
环境配置方法如下：
#### vibe
```shell
cd VIBE
source scripts/install_conda.sh
```
#### detectron2
```shell
cd detectron2
python -m pip install -e detectron2
```
#### openpose
```shell
conda create -n openpose python=3.7
conda activate openpose
cd pytorch-openpose
pip install -r requirements.txt
```

#### jppnet
这个环境官方没有提供环境信息，我们的代码中给出了jppnet.yml作为参考
```shell
conda env create -f jppnet.yml
```

#### humannerf
```shell
conda create --name humannerf python=3.7
conda activate humannerf
pip install -r requirements.txt
```

#### ctnet
我们的代码中给出了ctnet.yml作为参考
```shell
conda env create -f ctnet.yml
```

### Dataset Generation

#### wild data preparation
我们搭建了将数据从视频处理成人体重建所需数据格式的pipeline。
请准备好两段视频：Source Video和Target Video我们的工作将实现将Target Video中的人物姿态和服装迁移到Source Video中的人物上。
首先将两段视频放置到`./video`目录下。然后编辑`./process.sh`将data_id修改为您的视频名称。
这里我们建议将视频长度尽量控制在30s以内。

完成以上操作后，在当前目录执行
```shell
source process.sh
```
即可完成对视频的处理工作，处理后的数据集被保存在dataset/wild中的视频名称对应目录下。

#### zju_mocap
由于作者并未公开zju_mocap数据集，我们在此无法直接提供该数据集的内容，请在此处获取并下载[zju_mocap](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset)
将下载后的数据集放入./dataset/zju_mocap中即可

## Train Model
执行以下代码完成在自定义数据集上的训练
```shell
train_wild.sh
```
执行以下代码完成在zju\_mocap数据集上的训练
```shell
train_zju.sh
```
## Repose
执行以下代码完成在基于自定义数据集动作的姿势重建，人体模型以ckpt_path指定
```shell
repose_wild.sh
```
执行以下代码完成在基于zju_mocap数据集动作的姿势重建，人体模型以ckpt_path指定
```shell
repose_zju.sh
```

## ReCLothes
执行以下代码完成数据集的姿势、人体解析、IUV信息提取，请指定数据集地址和保存名称
```shell
source pipline.sh
```
执行完两组图片数据后，请执行以下代码完成服装更换任务：
```shell
source reclothes.sh
```
## Acknowledgement

我们的实现参考了 [HumanNERF](https://github.com/chungyiweng/humannerf), [Neuman](https://github.com/apple/ml-neuman),[CT-Net](https://github.com/yf1019/CT-Net)和[MUST-GAN](https://github.com/TianxiangMa/MUST-GAN). 我们十分感谢作者开源了他们的项目代码。