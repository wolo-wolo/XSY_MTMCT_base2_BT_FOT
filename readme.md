# XSY_MTMC T_baeline_2
Gaojian Wang

XSY_MTMCT_base_2主要基于base1进行代码优化，以实现不同场景的MTMC车辆跟踪。
该repository基于实验流程进行说明并简介相应module。
* 使用ByteTracker替换FairMot进行单镜头多目标跟踪(MOT/SCT)
* 添加轨迹过滤策略

## Requirements

Python 3.8 or later with all ```requirements.txt``` dependencies installed, including `torch>=1.7`. To install run:
```bash
$ pip install -r requirements.txt
$ cd detector/yolov7/
$ pip install -r requirements.txt
$ cd ../../datasets/Test_scene/eval/
$ pip install -r requirements.txt
```

## 1. 数据与模型准备 


### 1.1 AIC21(22)数据
AIC21数据(the data is same as AIC22-track-1-mtmc-vt).
从[AI City Challenge 21](https://www.aicitychallenge.org/2021-data-and-evaluation/)
下载Track 3：city-scale multi-target multi-camera vehicle tracking的数据AIC21_Track3_MTMC_Tracking，并将其放入文件夹```dataset```，目录结构如下：

>   * datasets
>     * AIC21_Track3_MTMC_Tracking
>       *  (unzip from AIC21_Track3_MTMC_Tracking.zip)

### 1.2 AIC21测试数据特征、检测模型、重识别模型
从[google drive link](https://drive.google.com/drive/folders/11616Gomc7MbjbgWrDruL26TGi9JNCAAE?usp=sharing)
下载：

1）detect_provided(不用下载，在后继实验生成)：提供的在AIC测试集S06的六个摄像头视频c041-c046所提取的detection和Re-ID特征

2）yolov5x.pt：在coco数据集上预训练的yolov5x检测模型

3）reid_model：AIC21-Track2(vehicle reid)[优胜方案](https://github.com/michuanhaohao/AICITY2021_Track2_DMT)
的三个reid模型，可以参照链接重新训练。

并以如下结构存放：
>   * datasets
>      * detect_provided (Including detection and corresponding Re-ID features)
>   * detector
>     * yolov5
>       * weights
>        * [yolov5x.pt](https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5x.pt) (Pre-trained yolov5x model on COCO)
>   * reid
>     * reid_model (Pre-trained reid model on Track 2)
>       * resnet101_ibn_a_2.pth
>       * resnet101_ibn_a_3.pth
>       * resnext101_ibn_a_2.pth


### 1.3 其它场景的测试数据
1. 例如一个场景下有三个视频：镜头1.mp4, 镜头2.mp4, 镜头3.mp4, 将所有视频重名为"vdo.mp4"并依次放入```dataset/Test_scene/test/```对应的目录下(S0X：场景编号任意。c00X：相机编号，推荐按序)。

2. 三个镜头之间可能存在bias(每个视频都有与开始时间的偏移量，可以用于同步)，记录在```dataset/Test_scene/cam_timestamp/S0X.txt```，如果未知则可置0：

S06.txt:
```
c001 0
c002 0
c003 0
```
3. 如果视频存在ROI区域mask图片roi.jpg(白色ROI，黑色ignore)，可一同放入。

4. 如果存在标注员使用Darklabel生成的标注，注意与视频镜头同名：c00X.csv

目录结构如下
>   * datasets
>      * Test_scene
>           * test
>               * S06
>                   * c001
>                       * vdo.mp4
>                       * roi.jpg（optional）
>                   * c002
>                       * vdo.mp4
>                       * roi.jpg（optional）
>                   * c003
>                       * vdo.mp4
>                       * roi.jpg（optional）
>           * cam_timestamp
>               * S06.txt
>           * test_labels
>               * S06
>                   * c001.csv
>                   * c001.csv
>                   * c001.csv

## 2. 配置文件修改

在```config```文件夹中，修改```aic_all.yml； aic_reid1.yml； aic_reid2.yml； aic_reid3.yml； aic_mcmt.yml```文件：
* 修改对应的路径目录
* 如果使用AIC21上的数据，将路径中```Test_scene```替换为```AIC21_Track3_MTMC_Tracking```。

**_`aic_all.yml:`_**
```
**准备的数据路径**
CHALLENGE_DATA_DIR: '/home/ubuntu/wgj/XSY_MTMCT_base1/datasets/Test_scene/'  
**从视频中提取的帧---gen_images_with_aic(label).py生成**
DET_SOURCE_DIR: '/home/ubuntu/wgj/XSY_MTMCT_base1/datasets/detection/images/test/S06/'  
**镜头c00x视频的：
yolov5x目标检测结果---gen_det.py生成dets，dets_debug, test_labels, c00X_dets.pkl。
三个模型融合的reid特征---merge_reid_feat.py生成c00X_dets_feat.pkl。
MOT结果---fair_app.py生成c00X_mot_feat_raw.pkl, c00X_mot.txt；
post_processing.py生成c002_mot_feat.pkl, res/c00X_mot.txt; 
trajectory_fusion.py生成使用zone策略过滤后的特征c00X_mot_feat_break.pkl。
**
DATA_DIR: '/home/ubuntu/wgj/XSY_MTMCT_base1/datasets/detect_merge/'  
REID_SIZE_TEST: [384, 384]    # 384, 256
# 视频的ROI mask
ROI_DIR: '/home/ubuntu/wgj/XSY_MTMCT_base1/datasets/Test_scene/test/S06/'  
# 镜头的时间bias
CID_BIAS_DIR: '/home/ubuntu/wgj/XSY_MTMCT_base1/datasets/Test_scene/cam_timestamp/'  
# 标注员使用Darklabel生成的标注，注意与镜头视频同名：C00X.csv
LABEL_DIR: '/home/ubuntu/wgj/XSY_MTMCT_base1/datasets/Test_scene/test_labels/'  
USE_RERANK: True
USE_FF: True
USE_ROI: False  # added by wgj
USE_ST_FILTER: True  # added by wgj
USE_CAMERA: False  # added by wgj
SCORE_THR: 0.1
# 最后产生的测试视频MTMC T结果
MCMT_OUTPUT_TXT: 'track3.txt'  
```

`aic_reid1.yml`;  `aic_reid2.yml`;  `aic_reid3.yml`
```
# 从视频中提取的帧（运行gen_images_with_aic(label).py后生成）
DET_SOURCE_DIR: '/home/ubuntu/wgj/XSY_MTMCT_base1/datasets/detection/images/test/S06/'
# reid模型，三个配置文件使用三个不同的reid模型
REID_MODEL: 'reid_model/resnet101_ibn_a_2.pth'
# reid网络
REID_BACKBONE: 'resnet101_ibn_a' 
DET_IMG_DIR: '/home/ubuntu/wgj/XSY_MTMCT_base1/datasets/detect_merge/'
# reidX模型生成的reid特征（extract_image_feat.py后生成）
DATA_DIR: '/home/ubuntu/wgj/XSY_MTMCT_base1/datasets/detect_reidX(1 or 2 or 3）/'
REID_SIZE_TEST: [384, 384]
```

`aic_mcmt.yml`类似


## 3. 其它配置修改
* ```label_preprocess.py```中的```mot_label_dir = './test_labels/S0X/'```和```df.to_csv(out_mtmct_label_dir + 'test_gt_S0X.txt'...)```修改为对应的S编号
* ```detector/yolov5/gen_det.sh```中的```seqs=(c002 c003 c004)```修改为对应的测试数据镜头编号
* ```reid/merge_reid_feat.py```中的```for cam in ['c002', 'c003', 'c004']```修改为对应的测试数据镜头编号
* ```tracker/MOTBaseline/run_aic.sh```中的```seqs=(c002 c003 c004)```修改为对应的测试数据镜头编号
* ```tracker/MOTBaseline/src/fair_app.py```中的```frame_rate```修改为对应的测试数据视频的fps(unified to 25, no need to change)
* ```reid/reid-matching/tools/trajectory_fusion.py```中的```scene_name = ['S06']```和```save_dir```修改为对应的S编号
* ```reid/reid-matching/tools/sub_cluster.py```中的```scene_name = ['S06']```和```scene_cluster = [[2, 3, 4]]```和```fea_dir```修改为对应的S与C的编号
* ```reid/reid-matching/tools/viz_mcmt.py```中的dir and file name
* * ```run_all``` 中```python eval.py test_gt_S06.txt ${MCMT_CONFIG_FILE}```的```test_gt_S06.txt```

## 4. 重现所有pipeline进行MTMC T
完成上述准备与修改后，运行脚本：
```
bash ./run_all.sh
```
生成的MOT/MTMC T可视化结果保存在```exp/viz```

生成的MTMC T结果(```config/aic_all.MCMT_OUTPUT_TXT```)：
该文本文件每行包含检测和跟踪车辆的详细信息，值以空格分隔，格式如下：

```
〈camera_id〉 〈obj_id〉 〈frame_id〉 〈xmin〉 〈ymin〉 〈width〉 〈height〉 〈xworld〉 〈yworld〉

〈camera_id〉 是摄像机数字ID，如c001-c00X，则介于1和X之间。
〈obj_id〉 是每个目标的数字ID。它应是一个正整数，并且对于跨多个摄像机中的每个目标ID都是一致的。
〈frame_id〉 表示当前视频中当前帧的帧数，从1开始！
〈xmin〉 〈ymin〉 〈width〉 〈height〉, 检测目标axis-aligned的矩形bbox，由其在图像画布内的像素值坐标表示，从图像的左上角计算。所有值都是整数。
〈xworld〉 〈yworld〉 是每个对象的投影底部点的GPS坐标。它们目前未用于评估，但将来可能会使用。因此，如有可能，将其包括在内将是有益的。
```

## 5. 文档与模块指导
docs/pipeline/workflow.md
