# Run all commands in one shell
.ONESHELL:

# Default target
.DEFAULT_GOAL := help

# Generic Variables
USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d")

# Fixed
HOME := /space/calico/1/users/Harsha
PROJ_DIR := $(shell pwd)
DATA_DIR := $(PROJ_DIR)/data
RESULTS_DIR := $(PROJ_DIR)/results/20220222/new-recons/
MODEL_DIR := $(PROJ_DIR)/models
SCRATCH_MODEL_DIR := /cluster/scratch/friday/for_harsha
ENV_DIR := $(HOME)/venvs

# Dynamic
ENV_NAME := synthseg-venv
CUDA_V := 10.1
PARAM_FILES_DIR = SynthSeg_param_files_manual_auto_photos_noCerebellumOrBrainstem
MODEL_NAME := S16R02n
CMD = sbatch --job-name=$(MODEL_NAME) submit.sh
# {echo | python | sbatch submit.sh}

ACTIVATE_ENV = source $(ENV_DIR)/$(ENV_NAME)/bin/activate
ACTIVATE_FS = source /usr/local/freesurfer/nmr-dev-env-bash

# variables for SynthSeg
labels_dir = $(DATA_DIR)/SynthSeg_label_maps_manual_auto_photos_noCerebellumOrBrainstem
MODEL_PATH = $(SCRATCH_MODEL_DIR)/$(MODEL_NAME)

# label maps parameters
generation_labels = $(DATA_DIR)/$(PARAM_FILES_DIR)/generation_charm_choroid_lesions.npy
neutral_labels = '5'
segmentation_labels = $(DATA_DIR)/$(PARAM_FILES_DIR)/segmentation_new_charm_choroid_lesions.npy
noisy_patches =

# output-related parameters
batch_size = 1
channels = 1
target_res =
output_shape = 160

# GMM-sampling parameters
generation_classes = $(DATA_DIR)/$(PARAM_FILES_DIR)/generation_classes_charm_choroid_lesions_gm.npy
prior_type = 'uniform'
prior_means =
prior_std =
# specific_stats = --specific_stats
# mix_prior_and_random = --mix_prior_and_random

# spatial deformation parameters
# no_flipping = --no_flipping
scaling =
rotation =
shearing =
translation = 
nonlin_std = (4, 0, 4)
nonlin_shape_factor = (0.0625, 0.0625, 0.0625)

# blurring/resampling parameters
# randomise_res = --randomise_res
data_res = (1, 16, 1)
thickness = (1, 0.001, 1)
downsample = --downsample
blur_range = 1.03

# bias field parameters
bias_std = .5
bias_shape_factor = (0.025, 0.0625, 0.025)
# same_bias_for_all_channels = --same_bias_for_all_channels

# architecture parameters
n_levels = 5           # number of resolution levels
conv_per_level = 2  # number of convolution per level
conv_size = 3          # size of the convolution kernel (e.g. 3x3x3)
unet_feat = 24   # number of feature maps after the first convolution
activation = 'elu'     # activation for all convolution layers except the last, which will use sofmax regardless
feat_mult = 2    # if feat_multiplier is set to 1, we will keep the number of feature maps constant throughout the
#                        network; 2 will double them(resp. half) after each max-pooling (resp. upsampling);
#                        3 will triple them, etc.

# Training parameters
lr = 1e-4               # learning rate
lr_decay = 0            # learning rate decay (knowing that Adam already has its own internal decay)
wl2_epochs = 1          # number of pre-training epochs with wl2 metric w.r.t. the layer before the softmax
dice_epochs = 100       # number of training epochs
steps_per_epoch = 2500  # number of iteration per epoch


.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<

.PHONY: list
list:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

remove-subject-copies:
	rm $(DATA_DIR)/SynthSeg_label_maps_manual_auto_photos_noCerebellumOrBrainstem/subject*copy*

create-subject-copies:
	python scripts/photos_utils.py

## training-default: Equivalent to running tutorials/3-training.py
training-default:
	$(ACTIVATE_ENV)
	export PYTHONPATH=$(PROJ_DIR)
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):/usr/pubsw/packages/CUDA/$(CUDA_V)/lib64

	python /autofs/space/calico_001/users/Harsha/SynthSeg/scripts/commands/training.py train\
			$(DATA_DIR)/training_label_maps \
			$(MODEL_DIR)/SynthSeg_training_BB_resume \
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

## training: Use this target to train/retrain(resume) custom models
training:
	$(ACTIVATE_ENV)
	export PYTHONPATH=$(PROJ_DIR)
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):/usr/pubsw/packages/CUDA/$(CUDA_V)/lib64
	
	$(CMD) $(PROJ_DIR)/scripts/commands/training.py train\
		$(labels_dir) \
		$(SCRATCH_MODEL_DIR)/$(MODEL_NAME) \
		\
		--generation_labels $(generation_labels) \
		--neutral_labels $(neutral_labels) \
		--segmentation_labels $(segmentation_labels) \
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
		--wl2_epochs $(wl2_epochs) \
		--dice_epochs $(dice_epochs) \
		--steps_per_epoch $(steps_per_epoch) \
		--message 'set n_neutral_labels on 20220401' \
		;

## resume-training: Use this target to resume training
resume-training:
	$(ACTIVATE_ENV)
	export PYTHONPATH=$(PROJ_DIR)
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):/usr/pubsw/packages/CUDA/$(CUDA_V)/lib64
	
	$(CMD) $(PROJ_DIR)/scripts/commands/training.py resume-train $(MODEL_PATH)

## predict: Inference using a trained model
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

predict-%: H5_FILE = /space/calico/1/users/Harsha/SynthSeg/models/S02R01/dice_100.h5
predict-%: RESULTS_DIR = $(PROJ_DIR)/results/20220222/new-recons/S02R01
# predict-%: H5_FILE = $(PROJ_DIR)/models/jei-model/SynthSegPhotos_no_brainstem_or_cerebellum_4mm.h5
predict-%: LABEL_LIST = $(PROJ_DIR)/models/jei-model/SynthSegPhotos_no_brainstem_or_cerebellum_4mm.label_list.npy
## predict-scans: Run MRI volumes through default SynthSeg
predict-scans:
	$(ACTIVATE_ENV)
	export PYTHONPATH=$(PROJ_DIR)
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):/usr/pubsw/packages/CUDA/10.1/lib64

	python $(PROJ_DIR)/scripts/commands/SynthSeg_predict.py \
		--i $(RESULTS_DIR)/mri.scans/ \
		--o $(RESULTS_DIR)/mri.synthseg/ \
		--vol $(RESULTS_DIR)/volumes/mri.synthseg.volumes

## predict-soft: Run soft recons through custom SynthSeg model
predict-soft:
	$(ACTIVATE_ENV)
	export PYTHONPATH=$(PROJ_DIR)
	export LD_LIBRARY_PATH=/usr/pubsw/packages/CUDA/10.1/lib64

	python scripts/commands/predict.py \
		--smoothing 0.5 \
		--biggest_component \
		--padding 256 \
		--vol $(RESULTS_DIR)/volumes/soft.synthseg.volumes \
		$(RESULTS_DIR)/soft.recon/ \
		$(RESULTS_DIR)/soft.synthseg/ \
		$(H5_FILE) \
		$(LABEL_LIST)

## predict-hard: Run hard recons through custom SynthSeg model
predict-hard:
	$(ACTIVATE_ENV)
	export PYTHONPATH=$(PROJ_DIR)
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):/usr/pubsw/packages/CUDA/10.1/lib64

	python scripts/commands/predict.py \
		--smoothing 0.5 \
		--biggest_component \
		--padding 256 \
		--vol $(RESULTS_DIR)/volumes/hard.synthseg.volumes \
		$(RESULTS_DIR)/hard.recon/ \
		$(RESULTS_DIR)/hard.synthseg/ \
		$(H5_FILE) \
		$(LABEL_LIST)


samseg-%: PROJ_DIR := /space/calico/1/users/Harsha/SynthSeg
samseg-%: DATA_DIR := $(PROJ_DIR)/data/uw_photo/Photo_data
samseg-%: SKIP := $(shell seq 1 4)
samseg-%: RESULTS_DIR := $(PROJ_DIR)/results/20220411/new-recons-skip
samseg-%: FSDEV = $(HOME)/photo-samseg-orig
samseg-%: ATL_FLAG := C0
# { C0 | C1 | C2 }
samseg-%: REF_KEY := hard
# { hard | soft | image }
# samseg-%: CMD := sbatch --job-name=$(REF_KEY)-new-$(ATL_FLAG)-skip-$$skip-$$sub_id submit-samseg.sh
# {echo | python | sbatch --job-name=$(REF_KEY)-new-$(ATL_FLAG)-skip-$$skip-$$sub_id submit-samseg.sh}
## samseg-new-recons: Run FS SAMSEG on new reconstructions
# This feature is available in FS, so we can do away with this target
samseg-new-recons:
	$(ACTIVATE_FS)
	export PYTHONPATH=$(FSDEV)/python/packages
	
	for i in `ls -d $(DATA_DIR)/*-*/`; do \
		for skip in $(SKIP); do \
			sub_id=`basename $$i`
			sbatch --job-name=$(REF_KEY)-new-$(ATL_FLAG)-skip-$$skip-$$sub_id submit-samseg.sh $(FSDEV)/python/scripts/run_samseg \
			-i $(DATA_DIR)/$$sub_id/ref_$(REF_KEY)_skip_$$skip/photo_recon.mgz \
			-o $(RESULTS_DIR)-$$skip/samseg_output_$(REF_KEY)_$(ATL_FLAG)/$$sub_id \
			--threads 64 \
			--dissection-photo both \
			--atlas $(FSDEV)/atlas; \
		done; \
	done

## samseg-hard-on-old-recons: Run FS SAMSEG on old hard reconstructions
# stupid mri_convert does not create directories for us and thus
# the use of mkdir (:grimace:)
samseg-hard-on-old-recons:
	$(ACTIVATE_FS)
	export PYTHONPATH=$(FSDEV)/python/packages
	
	for i in `ls -d /cluster/vive/UW_photo_recon/Photo_data/*-*/`; do \
		sub_id=`basename $$i`
		mkdir -p $(RESULTS_DIR)/SAMSEG_OUTPUT_HARD_C2/$$sub_id
		mri_convert /cluster/vive/UW_photo_recon/recons/results_Henry/Results_hard/$$sub_id/$$sub_id".hard.recon.mgz" $(RESULTS_DIR)/SAMSEG_OUTPUT_HARD_C2/$$sub_id/input.mgz
		sbatch submit-samseg.sh $(FSDEV)/python/scripts/run_samseg \
		-i $(RESULTS_DIR)/SAMSEG_OUTPUT_HARD_C2/$$sub_id/input.mgz \
		-o $(RESULTS_DIR)/SAMSEG_OUTPUT_HARD_C2/$$sub_id \
		--threads 64 \
		--dissection-photo both \
		--atlas $(FSDEV)/atlas; \
	done

## samseg-soft-on-old-recons: Run FS SAMSEG on old soft reconstructions
samseg-soft-on-old-recons:
	$(ACTIVATE_FS)
	export PYTHONPATH=$(FSDEV)/python/packages
	
	for i in `ls -d /cluster/vive/UW_photo_recon/Photo_data/*-*/`; do \
		sub_id=`basename $$i`
		sbatch submit-samseg.sh $(FSDEV)/python/scripts/run_samseg \
			-i /cluster/vive/UW_photo_recon/recons/results_Henry/Results_soft/$$sub_id/soft/$$sub_id"_soft.mgz" \
			-o $(RESULTS_DIR)/SAMSEG_OUTPUT_SOFT_C2/$$sub_id \
			--threads 64 \
			--dissection-photo both \
			--atlas $(FSDEV)/atlas; \
	done

## model_dice_map: Creates a csv file with the model name and the corresponding
# dice index necessitated by failed/early terminated training.
model_dice_map:
	python -c "from scripts import photos_utils; photos_utils.model_dice_map()"

## run_synthseg_inference: Run SynthSeg inference on all models
# - Takes dice_ids.csv as input
# - More on how this csv was generated can be found in scripts/photos_utils
# 	or refer to the make target: model_dice_map
run_synthseg_%: SKIP := $(shell seq 1 1)
run_synthseg_inference:
	out_dir=20220411
	for skip in $(SKIP); do \
		while IFS=, read -r model dice_idx _
		do
			sbatch --job-name=new-skip-$$skip-$$model \
			--export=ALL,model=$$model,dice_idx=$$dice_idx,out_dir=$$out_dir,script=new$$skip.py,idx=$$skip submit-pipeline.sh
		done < dice_ids$$skip.csv; \
	done

## just-plot: Plotting dice plots
# Necessitated because latex (via matplotlib) doesn't run on MLSC
just-plot:
	out_dir=20220404
	while IFS=, read -r model dice_idx
	do
		python $(PROJ_DIR)/scripts/hg_dice_scripts/new4.py --recon_flag 'new' --out_dir_name $$out_dir --model_name $$model --part 3;
	done < dice_ids4.csv	

## collect-plots: collect all dice images into a single pdf
collect-plots:
	out_dir=20220404
	folder_type=new
	python -c "from scripts import photos_utils; photos_utils.collect_images_into_pdf('$$out_dir/$$folder_type-recons-skip-4')"
