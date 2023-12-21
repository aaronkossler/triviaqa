#!/bin/bash
#SBATCH --ntasks=40
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/output.txt
#SBATCH --error=outputs/error.txt
#SBATCH --job-name=t5_pipeline
#SBATCH --mem=128000

pip install --upgrade pip
pip install -r ../requirements.txt

python pre_pipeline.py --model deepset/roberta-base-squad2 --domain web --gpu yes