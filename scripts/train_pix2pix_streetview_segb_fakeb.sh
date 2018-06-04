NAME=streetview_pix2pix_segb
MODEL_NAME=pix2pix
DATA_ROOT=./datasets/streetview_seg

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
    --generate_fake_b \
    --input_nc 7