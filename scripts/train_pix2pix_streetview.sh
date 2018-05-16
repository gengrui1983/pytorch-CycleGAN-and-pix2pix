Vanilla Pix2pix:
python train.py --dataroot ./datasets/new_streetview --name streetview_pix2pix --model pix2pix --which_model_netG resnet_6blocks_mlp --which_direction AtoB --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --no_ganFeat_loss

python test.py --dataroot ./datasets/new_streetview --name streetview_pix2pix --model pix2pix --which_model_netG resnet_6blocks_mlp --which_direction AtoB --dataset_mode aligned --norm batch

Pix2pix with Multiscale Discriminator and Feature Matching Loss:
python train.py --dataroot ./datasets/new_streetview --name streetview_pix2pix-multiscale-feature_matching_small --model pix2pix --which_model_netG resnet_6blocks_mlp --which_model_netD multi_scale --which_direction AtoB --lambda_A 100 --no_lsgan --dataset_mode aligned --norm batch --pool_size 0 --num_D 3

python test.py --dataroot ./datasets/new_streetview --name streetview_pix2pix-multiscale-feature_matching_small --model pix2pix --which_model_netG resnet_6blocks_mlp --which_direction AtoB --dataset_mode aligned --norm batch

Distance:
python train.py --dataroot ./datasets/new_streetview --name streetview_pix2pix-multiscale-feature_matching_small_dist --model pix2pix --which_model_netG resnet_6blocks_mlp --which_model_netD multi_scale --which_direction AtoB --lambda_A 100 --no_lsgan --dataset_mode aligned --norm batch --pool_size 0 --num_D 3 --use_dist --input_nc 4

python test.py --dataroot ./datasets/new_streetview --name streetview_pix2pix-multiscale-feature_matching_small_dist --model pix2pix --which_model_netG resnet_6blocks_mlp --which_direction AtoB --dataset_mode aligned --norm batch --use_dist --input_nc 4

python -m visdom.server