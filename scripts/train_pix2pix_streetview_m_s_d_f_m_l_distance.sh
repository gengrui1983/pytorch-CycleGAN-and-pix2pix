NAME=streetview_pix2pix
MODEL_NAME=pix2pix
DATA_ROOT=./datasets/new_streetview

python train.py \
    --dataroot ./datasets/new_streetview \
    --name streetview_pix2pix-multiscale-feature_matching_small_dist \
    --model pix2pix \
    --which_model_netG resnet_6blocks_mlp \
    --which_model_netD multi_scale \
    --which_direction AtoB \
    --lambda_A 100 \
    --no_lsgan \
    --dataset_mode aligned \
    --norm batch \
    --pool_size 0 \
    --num_D 3 \
    --use_dist \
    --input_nc 4