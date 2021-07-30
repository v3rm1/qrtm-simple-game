#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=vanilla-cartpole
#SBATCH --mem=8000
module restore thesis_mods

rm -f *.out

python3 ./run_simulations.py