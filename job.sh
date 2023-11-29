#!/bin/bash
#SBATCH --ntasks=40
#SBATCH --time=02:00:00
#SBATCH --mem=4000
#SBATCH --gres=gpu:1
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --job-name=t5_pipeline

pip install --upgrade pip
pip install -r requirements.txt

python notebooks/t5_pipeline.py