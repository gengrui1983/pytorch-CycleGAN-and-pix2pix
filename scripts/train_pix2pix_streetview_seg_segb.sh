NAME=streetview_pix2pix_seg_segb
MODEL_NAME=pix2pix
DATA_ROOT=./datasets/streetview_seg_car_in4

python train.py \
    --dataroot $DATA_ROOT \
    --name $NAME \
    --model $MODEL_NAME \
    --which_model_netG resnet_6blocks_mlp \
    --which_model_netD multi_scale \
    --which_direction AtoB \
    --lambda_A 100 \
    --no_lsgan \
    --dataset_mode aligned \
    --norm batch \
    --pool_size 0 \
    --num_D 3 \
    --for_seg \
    --use_dist \
    --for_segB \
    --input_nc 7