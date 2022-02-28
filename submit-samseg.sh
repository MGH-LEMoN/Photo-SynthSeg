#!/bin/bash
#SBATCH --account=lcnrtx
#SBATCH --partition=basic
##SBATCH --partition=dgx-a100,rtx8000,rtx6000,lcnrtx
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
##SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output="./logs/samseg/%x.out"
#SBATCH --error="./logs/samseg/%x.err"
#SBATCH --mail-user=hvgazula@umich.edu
#SBATCH --mail-type=FAIL

source /usr/local/freesurfer/nmr-dev-env-bash
export PHOTOSAMSEG=/space/calico/1/users/Harsha/photo-samseg-orig
export PYTHONPATH=$PHOTOSAMSEG/python/packages
export LD_LIBRARY_PATH=/usr/pubsw/packages/CUDA/10.1/lib64:/usr/pubsw/packages/CUDA/10.2/lib64

echo 'Start time:' `date`
echo "$@"
if [[ -v SLURM_ARRAY_TASK_ID ]]
then
    $PHOTOSAMSEG/python/bin/python3 "$@"
else
    $PHOTOSAMSEG/python/bin/python3 "$@"
fi
echo 'End time:' `date`