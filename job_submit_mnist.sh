#! /bin/bash

#SBATCH --job-name=mnist_train_10k
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=0-12:00
#SBATCH --mem=32G
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --open-mode=append

#SBATCH --mail-type=END

# rsync -ravx /n/home13/shubham/Current\ Projects/bioplausible_learning /n/holyscratch01/ba_lab/Users/shubham/
cd /n/home13/shubham/Current\ Projects/bioplausible_learning/code/notebooks/experiments/mnist

# cd /n/holyscratch01/ba_lab/Users/shubham/bioplausible_learning/code/notebooks/experiments/mnist
# cd 

source ~/.bashrc
mamba deactivate
mamba activate representations

python train_mnist.py

mamba deactivate


# rsync -ravx /n/holyscratch01/ba_lab/Users/shubham/bioplausible_learning /n/home13/shubham/Current\ Projects/


