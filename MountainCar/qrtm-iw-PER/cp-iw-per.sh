#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=iw-PER-cartpole
#SBATCH --mem=20000
module restore thesis_mods
rm -f *.out

python3 ./run_simulations.py