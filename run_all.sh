MCMT_CONFIG_FILE="aic_all.yml"

##### Preprocess label from XSY label maker.####
cd datasets/Test_scene/
#python label_preprocess.py

##### Run Detector.####
cd ../../detector/
#python gen_images_with_label.py ${MCMT_CONFIG_FILE}

#cd yolov5/
#bash gen_det.sh ${MCMT_CONFIG_FILE}
cd yolov7/
#bash gen_det.sh ${MCMT_CONFIG_FILE}

###### Extract reid feautres.###
cd ../../reid/
#python extract_image_feat.py "aic_reid1.yml"
#python extract_image_feat.py "aic_reid2.yml"
#python extract_image_feat.py "aic_reid3.yml"
#python merge_reid_feat.py ${MCMT_CONFIG_FILE}
######
########## MOT.(MOTBaseline is modified fairMOT) ####
##cd ../tracker/MOTBaseline/
##bash run_aic.sh ${MCMT_CONFIG_FILE}
#
cd ../tracker/ByteTrack/
#bash run_aic.sh ${MCMT_CONFIG_FILE}
#waitz
#
######### Get results. ####
cd ../../reid/reid-matching/tools
#python trajectory_fusion.py ${MCMT_CONFIG_FILE}
#python sub_cluster.py ${MCMT_CONFIG_FILE}
#python gen_res.py ${MCMT_CONFIG_FILE}
#python find_outlier_tracklet_v2.py ${MCMT_CONFIG_FILE}
##
############ Vis####
#####python viz_mot.py ${MCMT_CONFIG_FILE}
#python viz_mcmt.py ${MCMT_CONFIG_FILE}
###
###
############  eval metrics: IDF1 IDP IDR MOTA. #########
cd ../../../datasets/Test_scene/eval
#python eval.py test_gt_S06.txt ${MCMT_CONFIG_FILE}
##
####### Delete temporal results ####
cd ../../../
 # # rm -rf ./datasets/detection/
rm -rf ./datasets/detect_merge/
rm -rf ./datasets/detect_reid1/
rm -rf ./datasets/detect_reid2/
rm -rf ./datasets/detect_reid3/
rm -rf ./reid/reid-matching/tools/exp_ori/
rm -rf ./reid/reid-matching/tools/test_cluster.pkl
rm -rf ./tracker/ByteTrack/src/post_processing/cache
rm -rf ./tracker/ByteTrack/src/log.txt
rm -rf ./tracker/MOTBaseline/src/post_processing/cache
rm -rf ./tracker/MOTBaseline/src/log.txt
