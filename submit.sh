#!/bin/bash
#SBATCH --account=lcnrtx
#SBATCH --partition=rtx6000
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=3
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --job-name="hg824_test"
#SBATCH --output="synthseg_test.out"
#SBATCH --output="synthseg_test.err"

source /space/calico/1/users/Harsha/synthseg-venv/bin/activate
export PYTHONPATH=/space/calico/1/users/Harsha/synthseg-photos
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/pubsw/packages/CUDA/10.1/lib64

python3 /space/calico/1/users/Harsha/synthseg-photos/scripts/launch_training.py \
/space/calico/1/users/Harsha/synthseg-photos/data/SynthSeg_label_maps_manual_auto_photos_noCerebellumOrBrainstem \
/cluster/scratch/friday/4_hg824/synthseg_1 \
 --generation_labels /space/calico/1/users/Harsha/synthseg-photos/data/SynthSeg_param_files_manual_auto_photos_noCerebellumOrBrainstem/generation_charm_choroid_lesions.npy \
 --segmentation_labels /space/calico/1/users/Harsha/synthseg-photos/data/SynthSeg_param_files_manual_auto_photos_noCerebellumOrBrainstem/segmentation_new_charm_choroid_lesions.npy \
 --generation_classes /space/calico/1/users/Harsha/synthseg-photos/data/SynthSeg_param_files_manual_auto_photos_noCerebellumOrBrainstem/generation_classes_charm_choroid_lesions_gm.npy \
 --output_shape 160 \
 --randomise_res \
 --dice_epochs 300 \
 --wl2_epochs 0 \
 --data_res (1, 4, 1) \
 --thickness (1, 0.001, 1) \
 --n_channels 3 \
 --bias_shape_factor (0.025, 1, 0.025) \
 --nonlin_shape_factor (1/16, 1/4, 1/16) \
 --downsample 1 \
 --nonlin_std 4.