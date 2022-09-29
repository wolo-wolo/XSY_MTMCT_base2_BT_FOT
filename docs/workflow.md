本文档参照```run_all.sh```对该MTMC T的workflow和改动部分进行说明

```MCMT_CONFIG_FILE="aic_all.yml"```

```aic_all.yml(路径参考)```
```
CHALLENGE_DATA_DIR: '/home/ubuntu/wgj/XSY_MTMCT_base1/datasets/Test_scene/'
DET_SOURCE_DIR: '/home/ubuntu/wgj/XSY_MTMCT_base1/datasets/detection/images/test/S06/'
DATA_DIR: '/home/ubuntu/wgj/XSY_MTMCT_base1/datasets/detect_merge/'
REID_SIZE_TEST: [384, 384]    # 384, 256
ROI_DIR: '/home/ubuntu/wgj/XSY_MTMCT_base1/datasets/Test_scene/test/S06/'
CID_BIAS_DIR: '/home/ubuntu/wgj/XSY_MTMCT_base1/datasets/Test_scene/cam_timestamp/'
LABEL_DIR: '/home/ubuntu/wgj/XSY_MTMCT_base1/datasets/Test_scene/test_labels/'
USE_RERANK: True
USE_FF: True
USE_ROI: False  # added by wgj
USE_ST_FILTER: True  # adde
 by wgj
USE_CAMERA: False  # added by wgj
SCORE_THR: 0.1
MCMT_OUTPUT_TXT: 'track3.txt'
```

## Label preprocess
```
cd datasets/Test_scene/
python gen_images_with_aic.py ${MCMT_CONFIG_FILE}
```
将MOT标签转换为MTMCT标签，并修正公司内部打标软件的bbox溢出错误:
从```datasets/Test_scene/test_labels/```读取MOT标签，合并成MTMCT标签```test_gt.txt```至```datasets/Test_scene/eval/test_gt.txt```。



## 视频帧提取
```
cd detector/
python gen_images_with_aic.py ${MCMT_CONFIG_FILE}
```
从```CHALLENGE_DATA_DIR```视频中提取帧图像至```DET_SOURCE_DIR```:
1. 图像命名为'imgXXXXXX.jpg', XXXXXX为帧ID
2. USE_ROI: 原图中非ROI区域用0(黑)填充
3. (added by wgj)如果有测试视频的标签,可以替换为```gen_images_with_label.py```, 对每个视频仅提取标签的start与end FID之间的帧(最后跟踪的可视化结果也是该段)


## 目标检测
```
cd yolov5/
sh gen_det.sh ${MCMT_CONFIG_FILE}
(python detect2img.py --name ${seq} --weights weights/yolov5x.pt --conf 0.1 --agnostic --save-txt --save-conf --img-size 1280 --classes 2 5 7 --cfg_file $1&)
```

运行目标检测：
1. 生成如下文件/文件夹，在DATA_DIR```dataset/detect_merge/```。

```dets(dir)```: 检测到的目标图像(imgFrameID_objectID.jpg)

```dets_debug(dir)```: 画出目标bbox和label(类别+conf)的原图(imgFrameID.jpg)

```test_labels(dir)```: txt标签(cls, bbox(*xywh), conf) # xywh由xyxy转换并经过归一化

```c00X_dets.pkl(file)```:  检测特征，记录为如下字典

    # det_name: 如img000282_001是282帧中检测到的第2个目标
    out_dict[det_name] = 
        bbox: (x1, y1, x2, y2),             # 左上和右下坐标
        frame: p.stem,                          # (img000282)
        id: det_num,                             # 当前frame的目标ID (1)
        imgname: det_name.png,  # dets中对应的检测图像名称(img000282_001.png)
        class: det_class,                      # 类别，MS coco中car是2...
        conf: det_conf                         # confidence


2. 使用其它目标检测模型，如[yoloV7](https://github.com/WongKinYiu/yolov7)
(added by wgj)。对应的模型放入```detector/yolov7/weights/```， 并修改：

```cd yolov7/``` 进入对应的detector文件夹

```gen_det.sh```中调用detect2img.py的```--weights```参数为对应的目标检测器权重path，细节见detect2img.py





## REID
```
cd ../../reid/
python extract_image_feat.py "aic_reid1.yml"
python extract_image_feat.py "aic_reid2.yml"
python extract_image_feat.py "aic_reid3.yml"
python merge_reid_feat.py ${MCMT_CONFIG_FILE}

除了配置文件.yml，注意如下修改extract_image_feat.py:
BATCH_SIZE = 192  # 根据显存调整(192是4*2080Ti)
NUM_PROCESS = 4  # torch可调用的gpu数量
```

#### extract_image_feat.py:

从```datasets/detect_merge/c00X/dets```中加载检测到的目标图像，分别调用```reid/reid_model```中的三个reid模型提取reid特征。
然后reid特征与```datasets/detect_merge/c00x/c00X_dets.pkl```即检测特征进行合并，即字典中添加```'feat':2048-d Reid feature```，
获得(det+reid)特征```c00X_dets_feat.pkl```，根据Reid模型和镜头分别保存在```datasets/detect_reidX/c00X/```。

#### merge_reid_feat.py：

对三个reid模型获得的(Det+REID)特征进行合并，即对三个```c00X_dets_feat.pkl```中的‘feat'进行：L2正则化-mean，获得最终的(Det+REID)特征(仍命名为```c00X_dets_feat.pkl```)并保存至```/datasets/detect_merge/c000x/```。




## MOT
```
cd ../tracker/MOTBaseline
sh run_aic.sh ${MCMT_CONFIG_FILE}
```

####  fair_app.py:
1.从```datasets/detection```获取检测图像名称与size，从(det+reid)特征```c00X_dets_feat.pkl```
获取bbox和reid特征，经过NMS后以frame作为index记录为```bbox_dic[frame_index]```，```feat_dic[frame_index]```。

2.bbox_dic和feat_dic作为SCT(MOT)的输入（基于FairMOT，借用了JDE的track builder和track management，
即卡尔曼滤波器+casecade matching）。JDE tracker使用bbox_dic和feat_dic通过分配对应的Tracklet IDs：
```<track ID><bbox(tlwh)><Re-ID feature after smooth(feat/=l2(feat)>```。最后生成如下文件在```datasets/detect_merge/c00X/```:

```c00X_mot_feat_raw.pkl```:  MOT特征，记录为如下字典

    # image_name: {sequence_name}_{trackID}_{frameID}.png 如c002_1_279.png, c002_2_280.png, c002_3_282.png, c002_4_282.png,c002_5_284.png, c002_5_285.png
    mot_feat_dic[image_name] = 
        bbox: (x1, y1, x2, y2),             # 左上和右下坐标 by: tlwh->(t,l,t+w,l+h)
        frame: frame id,                          # img{int(fid):06d} 如img000282
        id: int(pid)                             # 当前frame的track ID
        imgname: det_name.png,  # dets中对应的检测图像名称(img000282_001.png)
        feat,                      # 2048-d ReID feature(after smooth)   

```c00X_mot.txt```: MOT result

```frame_idx, tid(track ID), t(x_min in bbox), l(y_min in bbox), w(weight in bbox), h(height in bbox), score, -1, -1, -1```

#### ./post_processing main.py:
根据MOT结果进行后处理：
1. 进行association.py: 对非重叠的不同TID的目标i，j两两之间计算ReID特征的余弦相似度，小于threshold则添加一条edge，
[len(tids)，len(edges)，Tid-i, Tid-j, similarity], 然后生成输入图解决
min cost perfect matching problem。在PA这一步，获得最终matched tracklets(在```tracker/MOTBaseline/src/post_processing/cache```)。

2. 通过remove len 1 track and interpolate减少FNs(default close)， 通过track_nms减少FPs。
在所有的pp(post-processing)后, merging N tracks，最后获得如下MOT特征和MOT结果在```datasets/detect_merge/c00X/```:


```c00X_mot_feat.pkl```: MOT特征after merged tracks，说明同上述的```c00X_mot_feat_raw.pkl```:

```'bbox': bbox, 'frame': frame, 'id': pid,'imgname': image_name, 'feat': feat```
***
```c00X_mot.txt```: MOT result after merged tracks:

```frame, pid, l, t, w, h, -1, -1, -1, -1```



## MTMCT
```
cd ../../reid/reid-matching/tools
python trajectory_fusion.py ${MCMT_CONFIG_FILE}
python sub_cluster.py ${MCMT_CONFIG_FILE}
python gen_res.py ${MCMT_CONFIG_FILE}
```
####  trajectory_fusion.py：
在```c00x_mot_feat.pkl```每个key([image_name])的value中(bbox,frame,...,feat)最后添加了‘zone’: None(default),保存在```datasets/detect_merge/c00X/c00X_mot_feat_break.pkl```。
如果使用了zone信息(USE_ZONE=TRUE, default is False)，会使用论文中提出的TFS和DBTM方法对MOT和bbox的结果进行过滤。
Then：
* 移除只出现一次的目标(Tid)
* 如果bbox面积小于500，移除。但是如果remove后的object<2，则不移除。
* 添加的新字段，IO_time = end_fid - start_fid
* 除了all_feat, 对所有相同Tid的feat计算mean_feat。

最后获得```c00X.pkl```save在```reid/reid-matching/tools/exp/viz/test/S06/trajectory/```，如下所示：

```
tid_data[tid] = {
'cam': cid,                # camera id(c00x中的X)
'tid': tid,                # tracklet id
'mean_feat': mean_feat,    # 所有相同Tid的mean_feat, 2048-d array
'zone_list': zone_list,    # default is None
'frame_list': frame_list,  # 出现该Tid的fid，如282，209,310,312....
'tracklet': 
{
'FID-i': {bbox': (x1, y1, x2, y2), 'frame': 'imgXXX', 'id': TID-i, 'imgname': 'c00X_{tID}_{fID}.png', 'feat': array([2048-d]), 'zone': None}}
'FID-j': {bbox': (x1, y1, x2, y2), 'frame': 'imgXXX', 'id': TID-i, 'imgname': 'c00X_{tID}_{fID}.png', 'feat': array([2048-d]), 'zone': None}}
...
} # FID即frame_list中的frame id；注意'tracklet'中所有的'id'与tid_data[tid]的tid均一致
'io_time': io_time  # IO_time = end_fid - start_fid
}
```
<Modified: add USE_ZONE to fit the case of with/without zones.>


####  sub_cluster.py：
* 根据```cid-tid```<如(2,3)第二个摄像头ID为3的>取```mean_feat```，对其normalize，可选st_mask进行过滤(基于进出方向与场景)，
以其作为visual cures进行rerank并计算相似度矩阵（余弦距离），使用k-reciprocal neighbors来更新矩阵。
* 然后进行层次聚类(Agglomerative clustering的complete linkage)。
获得```all_clu： [(cid,tid),(cid,tid)...], [(cid,tid),(cid,tid)...]```，
每个列表为一个clu/簇。
* 对于all_clu中，如果有clu的长度小于1（即只有一个<cid, tid>）则不作为簇，
最后结果```<camera-id,track-id>: cluster-id```保存在```reid/reid-matching/tools/test_cluster.pkl```。

<Modified: add USE_SAMERA to control the use of camera(make use of enter-exit flow based on camera).>


####  gen_res.py：
* 从```c00X_mot_feat_break.pkl```获取```img_rects[fid] = [[Tid, bbox(x1,y1,x2,y2)],...]```
* 计算新的bbox(default)：
```
cx = 0.5 * (x1+x2)
cy = 0.5 * (y1+y2)
h = min((y2-y1) * 1.2, (y2-y1) + 40) 
w = min((x2-x1) * 1.2, (x2-x1)  + 40)
x = cx + 0.5 * w
y = cy + 0.5 * h
```
or 使用原bbox: (x1, y1, x2-x1(w), y2-y1(h))

* 从```test_cluster.pkl```中即```[<camera-id,track-id>: cluster-id]```根据<cid, yid>以cluster-id作为new_tid。生成最终的MTMC T结果：
cid new_tid fid x y w h  -1 -1

<Modified: add USE_ROI to control the use of roi image.>

##eval metric
```
cd ../../../datasets/Test_scene/eval
python eval.py test_gt.txt ${MCMT_CONFIG_FILE}
```
评估测试指标IDF1 IDP IDR MOTA，其中
test_gt.txt: groud-truth label from label preprocess.

${MCMT_CONFIG_FILE}: from```${MCMT_CONFIG_FILE}```get```MCMT_OUTPUT_TXT```，即上一步生成的MTMC-T预测结果

## Vis
```
python viz_mot.py ${MCMT_CONFIG_FILE}
python viz_mcmt.py ${MCMT_CONFIG_FILE}
```
可视化结果