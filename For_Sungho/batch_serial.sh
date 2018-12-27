#!/bin/sh

## Some parameters for running a SLURM job
#SBATCH --job-name=GL_PYBREP
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=10G
#SBATCH --time=120:30:00
#SBATCH --ntasks=1
#SBATCH --input=none
#SBATCH --mail-user=sanghun-jee@oist.jp
#SBATCH --mail-type=FAIL,END
## Standard output and standard error files
#SBATCH --output=SHAREDDIR/pybrep.out.log
#SBATCH --error=SHAREDDIR/pybrep.err.log

module load intel/2016_update4 # use icc
module load intel.mpi/2016_update4 # use icc mpi

echo PYTHONPATH is $PYTHONPATH

python run_pyBrep.py

