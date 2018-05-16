NAME= streetview_pix2pix-multiscale-feature_matching_small
MODEL_NAME=pix2pix
DATA_ROOT=./datasets/new_streetview

python test.py \
    --dataroot $DATA_ROOT \
    --name $NAME \
    --model $MODEL_NAME \
    --which_model_netG resnet_6blocks_mlp \
    --which_direction AtoB \
    --dataset_mode aligned \
    --norm batch