#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=mlp_MC
#SBATCH --mem=10000
module restore thesis_mods

python3 ./q_mlp.py
