#!/bin/bash
rm -rf ./*.out
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=QRTM_MC
#SBATCH --mem=30000
module restore thesis_mods

python3 ./run_simulations.py