#!/bin/bash
#SBATCH --ntasks=40
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/output_pred.txt
#SBATCH --error=outputs/error_pred.txt
#SBATCH --job-name=t5_pipeline
#SBATCH --mem=128000

pip install --upgrade pip
pip install -r ../requirements.txt

python t5_predictions.py