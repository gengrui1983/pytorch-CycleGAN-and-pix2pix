NAME=streetview_pix2pix_segb_phase2
MODEL_NAME=pix2pix
DATA_ROOT=./datasets/new_streetview_with_seg

python test.py \
    --dataroot $DATA_ROOT \
    --name $NAME \
    --model $MODEL_NAME \
    --which_model_netG resnet_6blocks_mlp \
    --which_direction AtoB \
    --dataset_mode aligned \
    --norm batch \
    --for_seg \
    --how_many -1\
    --for_seg \
    --for_segB \
    --use_dist \
    --phase2 \
    --generate_fake_b \
    --loadSize 228 \
    --fineSize 200 \
    --input_nc 7