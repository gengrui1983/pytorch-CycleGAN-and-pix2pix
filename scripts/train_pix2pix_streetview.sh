NAME=streetview_pix2pix
MODEL_NAME=pix2pix
DATA_ROOT=./datasets/new_streetview

python train.py \
    --dataroot $DATA_ROOT \
    --name $NAME \
    --model $MODEL_NAME \
    --which_model_netG resnet_6blocks_mlp \
    --which_direction AtoB \
    --lambda_A 100 \
    --dataset_mode aligned \
    --no_lsgan \
    --norm batch \
    --pool_size 0 \
    --no_ganFeat_loss