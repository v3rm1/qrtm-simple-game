#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=MC_PER_QRTM
#SBATCH --mem=20000
module restore thesis_mods
python3 ./run_simulations.py
