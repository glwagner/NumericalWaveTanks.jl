#!/bin/bash
#SBATCH --job-name="linear instability"
#SBATCH --account=uri107
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH --output=%x.o%j.%N

julia --project linear_instability.jl
