NAME=streetview_pix2pix_segb_phase2
MODEL_NAME=pix2pix
DATA_ROOT=./datasets/new_streetview_with_seg

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
    --for_segB \
    --use_dist \
    --phase2 \
    --generate_fake_b \
    --loadSize 228 \
    --fineSize 200 \
    --input_nc 7