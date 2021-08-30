#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=MC_IW_PER
#SBATCH --mem=8000
module restore thesis_mods

python3 ./run_simulations.py