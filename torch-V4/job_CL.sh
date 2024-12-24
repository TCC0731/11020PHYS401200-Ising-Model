#! /bin/bash

#SBATCH --account MST113394
#SBATCH -p dev
#SBATCH -c 96
#SBATCH --nodes 1
#SBATCH --gpus-per-node 8

PATH=/home/u3518384/anaconda3/envs/py312-torch/bin:${PATH}
cd /home/u3518384/CompPhys/11020PHYS401200-Ising-Model/torch-V4

date
d1=$(date +"%s")
hostname
free -h

python -u Corr_len.py -d $1

date
d2=$(date +"%s")

echo $((d2-d1))