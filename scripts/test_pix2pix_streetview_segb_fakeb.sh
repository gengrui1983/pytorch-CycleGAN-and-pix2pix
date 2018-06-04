NAME=streetview_pix2pix_segb
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
    --for_seg \
    --use_dist \
    --how_many -1 \
    --for_segB \
    --generate_fake_b \
    --input_nc 7