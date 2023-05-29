#!/usr/bin/env bash
# File       : parallel_timing.sh
# Description: Job submission script for Team 19 parallel baseline timing
#SBATCH --job-name=parallel_timing
#SBATCH --output=team19_parallel_%j.out
#SBATCH --mem=8192
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00

module load gcc/12.1.0-fasrc01 openmpi/4.1.3-fasrc01

root=$(pwd -P)
cwd=parallel_check_${SLURM_JOBID}
mkdir -p ${cwd}
cd ${cwd}
cp -t . ${root}/*.h ${root}/*.cpp ${root}/Makefile ${root}/*.sh

make clean all

# change cpus-per-task above for number of OpenMP threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# run the programs:
# args: noprint smallestN largestN largestM step seed
#   if noprint==0, prints only dimensions and walltimes
#   if noprint==1, shows debug prints and correctness tests
# matrices start at M x N = smallestN x smallestN
#       and stop at M x N = largestM x largestN
# set seed for random matrix creation

srun ./parallel 0 256 512 512 256 333

# return the exit code of srun above
exit $?
