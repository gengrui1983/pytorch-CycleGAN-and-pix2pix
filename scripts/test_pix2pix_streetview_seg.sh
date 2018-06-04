NAME=streetview_pix2pix_seg
MODEL_NAME=pix2pix
DATA_ROOT=./datasets/streetview_seg

python test.py \
    --dataroot $DATA_ROOT \
    --name $NAME \
    --model $MODEL_NAME \
    --which_model_netG resnet_6blocks_mlp \
    --which_direction AtoB \
    --dataset_mode aligned \
    --norm batch \
    --use_dist \
    --for_seg \
    --how_many -1 \
    --input_nc 7