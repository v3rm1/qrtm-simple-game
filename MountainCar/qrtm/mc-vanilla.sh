#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=QRTM_MC
#SBATCH --mem=2000
module restore thesis_mods

python3 ./run_simulations.py