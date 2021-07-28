#!/bin/bash
#SBATCH --account=lcnrtx
#SBATCH --partition=rtx6000
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=3
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output="./logs/%A.out"
#SBATCH --error="./logs/%A.err"

# source /space/calico/1/users/Harsha/synthseg-venv/bin/activate
# export PYTHONPATH=/space/calico/1/users/Harsha/SynthSeg
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/pubsw/packages/CUDA/10.1/lib64

echo 'Requester:' $USER
echo 'Node:' $HOSTNAME
echo 'Start time:' `date`
echo "$@"
if [[ -v SLURM_ARRAY_TASK_ID ]]
then
    python "$@" --electrodes $SLURM_ARRAY_TASK_ID
else
    python "$@"
fi
echo 'End time:' `date`