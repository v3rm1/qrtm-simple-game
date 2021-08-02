#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=iw-PER-cartpole
#SBATCH --mem=2000
module restore thesis_mods
rm -f *.out

python3 ./run_simulations.py
