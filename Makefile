# Run all commands in one shell
.ONESHELL:

# Default target
.DEFAULT_GOAL := help

# Generic Variables
USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d")

PROJ_DIR := $(shell pwd)
DATA_DIR := $(PROJ_DIR)/data
RESULTS_DIR := $(PROJ_DIR)/results
# MODEL_DIR := $(PROJ_DIR)/models
CMD = python
# {echo | python | sbatch submit.sh}

ACTIVATE_ENV = source /space/calico/1/users/Harsha/venvs/synthseg-venv/bin/activate

# variables for SynthSeg
labels_dir = /space/calico/1/users/Harsha/SynthSeg/data/SynthSeg_label_maps_manual_auto_photos_noCerebellumOrBrainstem
model_dir = /cluster/scratch/friday/for_harsha/test

## label maps parameters ##
generation_labels = $(DATA_DIR)/SynthSeg_param_files_manual_auto_photos_noCerebellumOrBrainstem/generation_charm_choroid_lesions.npy
segmentation_labels = $(DATA_DIR)/SynthSeg_param_files_manual_auto_photos_noCerebellumOrBrainstem/segmentation_new_charm_choroid_lesions.npy
noisy_patches =

## output-related parameters ##
batch_size = 1
channels = 1
target_res =
output_shape = 192

# GMM-sampling parameters
generation_classes = $(DATA_DIR)/SynthSeg_param_files_manual_auto_photos_noCerebellumOrBrainstem/generation_classes_charm_choroid_lesions_gm.npy
prior_type = 'uniform'
prior_means =
prior_std =
# specific_stats = --specific_stats
# mix_prior_and_random = --mix_prior_and_random

## spatial deformation parameters ##
# no_flipping = --no_flipping
scaling =
rotation =
shearing =
translation = 
nonlin_std = 3
nonlin_shape_factor = (0.04, 0.25, 0.04)

## blurring/resampling parameters ##
# randomise_res = --randomise_res
data_res = (1, 4, 1)
thickness = (1, 0.01, 1)
downsample = --downsample
blur_range = 1.03

## bias field parameters ##
bias_std = .5
bias_shape_factor = (0.04, 0.25, 0.04)
# same_bias_for_all_channels = --same_bias_for_all_channels

## architecture parameters ##
n_levels = 5           # number of resolution levels
conv_per_level = 2  # number of convolution per level
conv_size = 3          # size of the convolution kernel (e.g. 3x3x3)
unet_feat = 24   # number of feature maps after the first convolution
activation = 'elu'     # activation for all convolution layers except the last, which will use sofmax regardless
feat_mult = 2    # if feat_multiplier is set to 1, we will keep the number of feature maps constant throughout the
#                        network; 2 will double them(resp. half) after each max-pooling (resp. upsampling);
#                        3 will triple them, etc.

## Training parameters ##
lr = 1e-4               # learning rate
lr_decay = 0            # learning rate decay (knowing that Adam already has its own internal decay)
wl2_epochs = 1          # number of pre-training epochs with wl2 metric w.r.t. the layer before the softmax
dice_epochs = 10       # number of training epochs
steps_per_epoch = 5  # number of iteration per epoch
# checkpoint = '20211004-model' 		# checkpoint name

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: list
list:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

remove-subject-copies:
	rm $(DATA_DIR)/SynthSeg_label_maps_manual_auto_photos_noCerebellumOrBrainstem/subject*copy*

create-subject-copies:
	python scripts/photos_utils.py

resume-training:
	$(ACTIVATE_ENV)
	export PYTHONPATH=$(PROJ_DIR)
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):/usr/pubsw/packages/CUDA/10.1/lib64
	
	python $(PROJ_DIR)/scripts/commands/training.py resume-train /cluster/scratch/friday/for_harsha/test-668939/


# Running this target is equivalent to running tutorials/3-training.py
training-default:
	$(ACTIVATE_ENV)
	export PYTHONPATH=$(PROJ_DIR)
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):/usr/pubsw/packages/CUDA/10.1/lib64

	python /autofs/space/calico_001/users/Harsha/SynthSeg/scripts/commands/training.py train\
			/space/calico/1/users/Harsha/SynthSeg/data/training_label_maps \
			/space/calico/1/users/Harsha/SynthSeg/models/SynthSeg_training_BB_resume \
			\
			--generation_labels $(DATA_DIR)/labels_classes_priors/generation_labels.npy 		\
			--segmentation_labels $(DATA_DIR)/labels_classes_priors/segmentation_labels.npy 	\
			--batch_size 1 			\
			--channels 1 			\
			--target_res  			\
			--output_shape 96 		\
			--generation_classes $(DATA_DIR)/labels_classes_priors/generation_classes.npy 		\
			--prior_type 'uniform' 	\
			--scaling .15 			\
			--rotation 15 			\
			--shearing .012 		\
			--translation  			\
			--nonlin_std '3' 		\
			--randomise_res 		\
			--blur_range 1.03 		\
			--bias_std .5 			\
			--n_levels 5            \
			--conv_per_level 2   	\
			--conv_size 3           \
			--unet_feat 24    		\
			--feat_mult 2     		\
			--activation 'elu'      \
			--lr 1e-4               \
			--lr_decay 0            \
			--wl2_epochs 1          \
			--dice_epochs 10       	\
			--steps_per_epoch 5   	\
			;


# Running this target is equivalent to resuming training using a model trained in the above target
resume-training-default:
	$(ACTIVATE_ENV)
	export PYTHONPATH=$(PROJ_DIR)
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):/usr/pubsw/packages/CUDA/10.1/lib64

	python $(PROJ_DIR)/scripts/commands/training.py resume-train 	\
			$(DATA_DIR)/training_label_maps 						\
			$(PROJ_DIR)/models/SynthSeg_training_BB_resume 			\
			\
			--generation_labels $(DATA_DIR)/labels_classes_priors/generation_labels.npy 			\
			--segmentation_labels $(DATA_DIR)/labels_classes_priors/segmentation_labels.npy 		\
			--batch_size 1 			\
			--channels 1 			\
			--target_res  			\
			--output_shape 96 		\
			--generation_classes $(DATA_DIR)/labels_classes_priors/generation_classes.npy 			\
			--prior_type 'uniform' 	\
			--scaling .15 			\
			--rotation 15 			\
			--shearing .012 		\
			--translation  			\
			--nonlin_std '3' 		\
			--randomise_res 		\
			--blur_range 1.03 		\
			--bias_std .5 			\
			--n_levels 5            \
			--conv_per_level 2   	\
			--conv_size 3           \
			--unet_feat 24    		\
			--feat_mult 2     		\
			--activation 'elu'      \
			--lr 1e-4               \
			--lr_decay 0            \
			--wl2_epochs 0          \
			--dice_epochs 10       	\
			--steps_per_epoch 5   	\
			--checkpoint /space/calico/1/users/Harsha/SynthSeg/models/SynthSeg_training_BB-665785/dice_005.h5                				\
			;


# This is the target that should be used to train/resume training models
training:
	$(ACTIVATE_ENV)
	export PYTHONPATH=$(PROJ_DIR)
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):/usr/pubsw/packages/CUDA/10.1/lib64
	
	$(CMD) $(PROJ_DIR)/scripts/commands/training.py train\
		$(labels_dir) \
		$(model_dir) \
		\
		--generation_labels $(generation_labels) \
		--segmentation_labels $(segmentation_labels) \
		--noisy_patches $(noisy_patches) \
		\
		--batch_size $(batch_size) \
		--channels $(channels) \
		--target_res $(target_res) \
		--output_shape $(output_shape) \
		\
		--generation_classes $(generation_classes) \
		--prior_type $(prior_type) \
		--prior_means $(prior_means) \
		--prior_std $(prior_std) \
		$(specific_stats) \
		$(mix_prior_and_random) \
		\
		$(no_flipping) \
		--scaling $(scaling) \
		--rotation $(rotation) \
		--shearing $(shearing) \
		--translation $(translation) \
		--nonlin_std '$(nonlin_std)' \
		--nonlin_shape_factor '$(nonlin_shape_factor)' \
		\
		$(randomise_res) \
		--data_res '$(data_res)' \
		--thickness '$(thickness)' \
		$(downsample) \
		--blur_range $(blur_range) \
		\
		--bias_std $(bias_std) \
		--bias_shape_factor '$(bias_shape_factor)' \
		$(same_bias_for_all_channels) \
		\
		--n_levels $(n_levels) \
		--conv_per_level $(conv_per_level) \
		--conv_size $(conv_size) \
		--unet_feat $(unet_feat) \
		--feat_mult $(feat_mult) \
		--activation $(activation) \
		\
		--lr $(lr) \
		--lr_decay $(lr_decay) \
		--wl2_epochs 0 \
		--dice_epochs $(dice_epochs) \
		--steps_per_epoch $(steps_per_epoch) \
		--message 'New training on 20211004' \
		--checkpoint /cluster/scratch/friday/for_harsha/test-668939/dice_005.h5 \
		;

predict:
	$(ACTIVATE_ENV)
	export PYTHONPATH=$(PROJ_DIR)
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):/usr/pubsw/packages/CUDA/10.1/lib64

	$(CMD) $(PROJ_DIR)/scripts/commands/predict.py
	--model /cluster/scratch/friday/models/test_photos_no_brainstem_or_cerebellum/dice_038.h5 \
	--label_list /space/calico/1/users/Harsha/SynthSeg/data/SynthSeg_param_files_manual_auto_photos_noCerebellumOrBrainstem//segmentation_new_charm_choroid_lesions.npy \
	--smoothing 0.5
	--biggest_component \
	--out_seg /tmp/seg4mm.mgz  /cluster/vive/UW_photo_recon/recons/results_Henry/Results_hard/17-0333/17-0333.hard.recon.grayscale.mgz

predict1:
	$(ACTIVATE_ENV)
	export PYTHONPATH=$(PROJ_DIR)
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):/usr/pubsw/packages/CUDA/10.1/lib64

	$(CMD) $(PROJ_DIR)/scripts/commands/predict.py \
		--model /cluster/scratch/friday/for_harsha/20210819-436612/dice_076.h5 \
		--label_list /space/calico/1/users/Harsha/SynthSeg/data/SynthSeg_param_files_manual_auto_photos_noCerebellumOrBrainstem/segmentation_new_charm_choroid_lesions.npy \
		--out_seg /space/calico/1/users/Harsha/SynthSeg/results/UW_photos/segmentations-latest-192/ \
		--topology_classes /space/calico/1/users/Harsha/SynthSeg/data/SynthSeg_param_files_manual_auto_photos_noCerebellumOrBrainstem/topo_classes.npy \
		--smoothing 0.5 \
		--biggest_component \
		/space/calico/1/users/Harsha/SynthSeg/results/UW_photos/
		;

predict-scans:
	$(ACTIVATE_ENV)
	export PYTHONPATH=$(PROJ_DIR)
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):/usr/pubsw/packages/CUDA/10.1/lib64

	python $(PROJ_DIR)/scripts/commands/SynthSeg_predict.py \
		--i /space/calico/1/users/Harsha/SynthSeg/results/UW.photos.mri.scans \
		--o /space/calico/1/users/Harsha/SynthSeg/results/UW.photos.mri.scans.segmentations/ \
		--vol /space/calico/1/users/Harsha/SynthSeg/results/UW.photos.mri.scans.segmentations/

predict-soft:
	$(ACTIVATE_ENV)
	export PYTHONPATH=$(PROJ_DIR)
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):/usr/pubsw/packages/CUDA/10.1/lib64

	python scripts/commands/predict.py \
		--smoothing 0.5 \
		--biggest_component \
		--padding 256 \
		--vol /space/calico/1/users/Harsha/SynthSeg/results/UW.photos.soft.recon.segmentations.jei \
		/space/calico/1/users/Harsha/SynthSeg/results/UW.photos.soft.recon/ \
		/space/calico/1/users/Harsha/SynthSeg/results/UW.photos.soft.recon.segmentations.jei/ \
		/space/calico/1/users/Harsha/4harsha/SynthSegPhotos_no_brainstem_or_cerebellum_4mm.h5 \
		/space/calico/1/users/Harsha/4harsha/SynthSegPhotos_no_brainstem_or_cerebellum_4mm.label_list.npy

predict-hard:
	$(ACTIVATE_ENV)
	export PYTHONPATH=$(PROJ_DIR)
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):/usr/pubsw/packages/CUDA/10.1/lib64

	python scripts/commands/predict.py \
		--smoothing 0.5 \
		--biggest_component \
		--padding 256 \
		--vol /space/calico/1/users/Harsha/SynthSeg/results/UW.photos.hard.recon.segmentations.jei \
		/space/calico/1/users/Harsha/SynthSeg/results/UW.photos.hard.recon/ \
		/space/calico/1/users/Harsha/SynthSeg/results/UW.photos.hard.recon.segmentations.jei/ \
		/space/calico/1/users/Harsha/4harsha/SynthSegPhotos_no_brainstem_or_cerebellum_4mm.h5 \
		/space/calico/1/users/Harsha/4harsha/SynthSegPhotos_no_brainstem_or_cerebellum_4mm.label_list.npy