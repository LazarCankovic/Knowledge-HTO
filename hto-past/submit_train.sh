#!/bin/bash
#SBATCH --job-name=hto_four_towers
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Create logs dir if it doesn't exist
mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# Set PYTHONPATH and run training
export PYTHONPATH=/fs/atipa/data/nsf-jg/hto/hto2/src:$PYTHONPATH

module laod python3/3.12.12
module load ml/torch/2.6
python -u src/hto/train_four_towers_timeaware_allcodes_weights.py

echo "End: $(date)"
