#!/bin/bash
#SBATCH -p gpuqs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -J hto_train
#SBATCH -o hto_train_%j.out

cd /fs/atipa/data/nsf-jg/hto/hto2
export PYTHONPATH=/fs/atipa/data/nsf-jg/hto/hto2/src:$PYTHONPATH

python src/hto/train_four_towers_timeaware_allcodes.py
