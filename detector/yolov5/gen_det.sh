#seqs=(c001 c002 c003 c004 c005 c006)   # test/s0
#gpu_id=0
#gpu_last_id=3

#if <4 camera, markdown above and use below:
seqs=(c001 c002 c003)   # test/s06
gpu_id=0 # test/s06
gpu_last_id=2 # test/s06

for seq in ${seqs[@]}
do
    echo "$seq -- Using GPU:$gpu_id"
    CUDA_VISIBLE_DEVICES=${gpu_id} python detect2img.py \
        --name ${seq} \
        --weights ./weights/yolov5x.pt \
        --conf 0.1 \
        --agnostic \
        --save-txt \
        --save-conf \
        --img-size 1280 \
        --classes 2 5 7 \
        --cfg_file $1&

    if [[ $gpu_id = $gpu_last_id ]]; then
        gpu_id=0
    else
        gpu_id=$(($gpu_id+1))
    fi

done
wait
