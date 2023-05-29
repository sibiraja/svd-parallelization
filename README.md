# TEAM 19
## Parallelization of SVD

Repository stucture:

- milestone: contains milestone docs (presentations, report, etc)
- parallel_src: contains the source code and job scripts for the parallel implementation
- sequential_src: contains the source code and job scripts for the sequential baseline

How to run locally:

1. Change into the directory of the chosen implementation (sequential or parallel)
2. Call the commands
```
make clean sequential
```
or 
```
make clean parallel
```
to compile
3. Run the program with the following arguments
```
./sequential <noprint> <smallest N> <largest N> <largest M> <step> <seed>
./parallel <noprint> <smallest N> <largest N> <largest M> <step> <seed>
# args descriptions:
#   if noprint==0, prints only dimensions and walltimes
#   if noprint==1, shows debug prints and visual correctness test
#   matrices start at M x N = smallestN x smallestN
#       and stop at M x N = largestM x largestN
#   set seed for random matrix creation
```

How to run on the academic cluster:
1. Change into the directory of the chosen implementation (sequential or parallel)
2. Call the commands
```
sbatch baseline_timing.sh
```
to submit a sequential baseline SLURM job or
```
sbatch parallel_timing.sh
```
to submit a parallel SLURM job. For the parallel job, edit the relevant #SBATCH lines to set an amount of cores or memory.