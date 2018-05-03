python train.py --dataroot ./datasets/new_streetview --name streetview_pix2pix --model pix2pix --which_model_netG resnet_6blocks_mlp --which_direction AtoB --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0

python test.py --dataroot ./datasets/new_streetview --name streetview_pix2pix-multiscale-feature_matching_small --model pix2pix --which_model_netG resnet_6blocks_mlp --which_direction AtoB --dataset_mode aligned --norm batch

python train.py --dataroot ./datasets/new_streetview --name streetview_pix2pix-multiscale-feature_matching_small --model pix2pix --which_model_netG resnet_6blocks_mlp --which_model_netD multi_scale --which_direction AtoB --lambda_A 100 --no_lsgan --dataset_mode aligned --norm batch --pool_size 0 --no_ganFeat_loss --num_D 2

python -m visdom.server