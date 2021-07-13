#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=PER_qrtm_cartpole
#SBATCH --mem=2000
module restore thesis_mods
python3 ~/../../data/s3893030/qrtm-simple-game/mountaincar/qrtm-iw/mc-iw.sh
