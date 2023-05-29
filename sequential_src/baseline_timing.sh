#!/usr/bin/env bash
# File       : baseline_timing.sh
# Description: Job submission script for Team 19 sequential baseline timing
#SBATCH --job-name=baseline_timing
#SBATCH --output=cs205_team19_%j.out
#SBATCH --mem=8192
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00

module load gcc/12.1.0-fasrc01

root=$(pwd -P)
cwd=sequential_check_${SLURM_JOBID}
mkdir -p ${cwd}
cd ${cwd}
cp -t . ${root}/*.cpp ${root}/Makefile ${root}/*.sh

make clean all

# run the programs:
# args: noprint smallestN largestN largestM step seed
# matrices start at M x N = smallestN x smallestN
#       and stop at M x N = largestM x largestN
# srun ./sequential 0 768 768 768 512
# set seed for random matrix creation
srun ./sequential 0 256 512 512 256 333

# return the exit code of srun above
exit $?
