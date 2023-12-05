#!/bin/bash
#SBATCH --ntasks=40
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/output_pipe.txt
#SBATCH --error=outputs/error_pipe.txt
#SBATCH --job-name=t5_pipeline
#SBATCH --mem=128000

pip install --upgrade pip
pip install -r ../requirements.txt

python t5_pipeline.py --batch_size 32 --tokenizer google/flan-t5-base --model google/flan-t5-base