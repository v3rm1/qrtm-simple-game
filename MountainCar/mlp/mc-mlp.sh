#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=mlp_MC
#SBATCH --mem=5000
module restore thesis_mods

rm -rf *.out

python3 ./q_mlp.py
