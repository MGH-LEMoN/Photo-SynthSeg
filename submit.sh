#!/bin/bash
#SBATCH --account=lcnrtx
#SBATCH --partition=rtx8000,rtx6000
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output="./logs/%A-%x.out"
#SBATCH --error="./logs/%A-%x.err"
#SBATCH --mail-user=hvgazula@umich.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT_90,END

source /space/calico/1/users/Harsha/venvs/synthseg-venv/bin/activate
export PYTHONPATH=/space/calico/1/users/Harsha/SynthSeg
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/pubsw/packages/CUDA/10.1/lib64

echo 'Start time:' `date`
echo "$@"
if [[ -v SLURM_ARRAY_TASK_ID ]]
then
    python "$@" --electrodes $SLURM_ARRAY_TASK_ID
else
    python "$@"
fi
echo 'End time:' `date`