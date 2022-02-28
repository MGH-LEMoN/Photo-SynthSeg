#!/bin/bash
## '''
## This shell script is used to run the entire pipeline from gathering all
## recontructions, segmentations to generating volume correlations and dice plots
## '''
#SBATCH --account=lcnrtx
#SBATCH --partition=rtx6000,rtx8000,lcnrtx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=96G
#SBATCH --time=0-01:30:00
#SBATCH --output="./logs/synth-infer/%x.out"
#SBATCH --error="./logs/synth-infer/%x.err"
#SBATCH --open-mode=append
#SBATCH --mail-user=hvgazula@umich.edu
#SBATCH --mail-type=FAIL

source /space/calico/1/users/Harsha/venvs/synthseg-venv/bin/activate
export PYTHONPATH=/space/calico/1/users/Harsha/SynthSeg
export LD_LIBRARY_PATH=/usr/pubsw/packages/CUDA/10.1/lib64:/usr/pubsw/packages/CUDA/10.2/lib64

export PROJ_DIR=/space/calico/1/users/Harsha/SynthSeg
export H5_FILE=/space/calico/1/users/Harsha/SynthSeg/models/$model/dice_$dice_idx.h5
export RESULTS_DIR=$PROJ_DIR/results/$out_dir/new-recons/$model
export LABEL_LIST=$PROJ_DIR/models/jei-model/SynthSegPhotos_no_brainstem_or_cerebellum_4mm.label_list.npy

export CMD=python

$CMD $PROJ_DIR/scripts/hg_dice_scripts/new.py \
    --recon_flag 'new' \
    --out_dir_name $out_dir \
    --model_name $model \
    --part 1

## predict-scans: Run MRI volumes through default SynthSeg
$CMD $PROJ_DIR/scripts/commands/SynthSeg_predict.py \
    --i $RESULTS_DIR/mri.scans/ \
    --o $RESULTS_DIR/mri.synthseg/ \
    --vol $RESULTS_DIR/volumes/mri.synthseg.volumes

## predict-soft: Run soft recons through custom SynthSeg model
$CMD $PROJ_DIR/scripts/commands/predict.py \
    --smoothing 0.5 \
    --biggest_component \
    --padding 256 \
    --vol $RESULTS_DIR/volumes/soft.synthseg.volumes \
    $RESULTS_DIR/soft.recon/ \
    $RESULTS_DIR/soft.synthseg/ \
    $H5_FILE \
    $LABEL_LIST

## predict-hard: Run hard recons through custom SynthSeg model
$CMD $PROJ_DIR/scripts/commands/predict.py \
    --smoothing 0.5 \
    --biggest_component \
    --padding 256 \
    --vol $RESULTS_DIR/volumes/hard.synthseg.volumes \
    $RESULTS_DIR/hard.recon/ \
    $RESULTS_DIR/hard.synthseg/ \
    $H5_FILE \
    $LABEL_LIST

$CMD $PROJ_DIR/scripts/hg_dice_scripts/new.py --recon_flag 'new' --out_dir_name $out_dir --model_name $model --part 2
