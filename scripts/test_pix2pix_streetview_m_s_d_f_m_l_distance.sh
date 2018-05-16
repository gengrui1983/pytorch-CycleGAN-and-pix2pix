NAME=streetview_pix2pix
MODEL_NAME=pix2pix
DATA_ROOT=./datasets/new_streetview

python test.py \
    --dataroot ./datasets/new_streetview \
    --name streetview_pix2pix-multiscale-feature_matching_small_dist \
    --model pix2pix \
    --which_model_netG resnet_6blocks_mlp \
    --which_direction AtoB \
    --dataset_mode aligned \
    --norm batch \
    --use_dist \
    --input_nc 4