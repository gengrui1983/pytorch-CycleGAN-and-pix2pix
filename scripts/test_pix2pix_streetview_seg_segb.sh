NAME=streetview_pix2pix_seg_segb
MODEL_NAME=pix2pix
DATA_ROOT=./datasets/streetview_seg_car_in4

python test.py \
    --dataroot $DATA_ROOT \
    --name $NAME \
    --model $MODEL_NAME \
    --which_model_netG resnet_6blocks_mlp \
    --which_direction AtoB \
    --dataset_mode aligned \
    --norm batch \
    --for_seg \
    --use_dist \
    --how_many -1 \
    --for_segB \
    --input_nc 7