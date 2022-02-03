#!/bin/bash
#SBATCH --account=lcnrtx
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2  
##SBATCH --ntasks=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output="./logs/hard-%A.out"
#SBATCH --error="./logs/hard-%A.err"

source /usr/local/freesurfer/nmr-dev-env-bash
export PHOTOSAMSEG=/space/calico/1/users/Harsha/photo-samseg-orig
export PYTHONPATH=$PHOTOSAMSEG/python/packages

echo 'Start time:' `date`
echo "$@"
if [[ -v SLURM_ARRAY_TASK_ID ]]
then
    $PHOTOSAMSEG/python/bin/python3 "$@"
else
    $PHOTOSAMSEG/python/bin/python3 "$@"
fi
echo 'End time:' `date`