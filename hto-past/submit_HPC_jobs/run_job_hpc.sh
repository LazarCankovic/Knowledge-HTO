
#!/bin/bash
#SBATCH --job-name=bert_train
#SBATCH --output=bert.out
#SBATCH --partition=gpuqs
#SBATCH --nodelist=cs[001-004]
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

cd /fs/atipa/data/nsf-jg/hto/models
source myenv/bin/activate

export CUDA_VISIBLE_DEVICES=2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python model1.py

