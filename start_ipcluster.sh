#!/bin/bash

#SBATCH --job-name=ipcluster
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=8g
#SBATCH --ntasks=12
#SBATCH --nodes=1-1

#SBATCH --mail-user=ines.wichert@oist.jp
#SBATCH --mail-type=FAIL

#SBATCH --output=%j.out.log
#SBATCH --error=%j.err.log

# Here we activate pybrep with python 3.6 via anaconda
# export PATH=/apps/unit/DeSchutterU/anaconda/bin:$PATH  # use anaconda for python
# source activate pybrep

# Could not install anything, so I cloned the environment.
export PATH=/home/i/ines-wichert/.conda/envs/ines_pybrep:$PATH
source activate ines_pybrep
export PYTHONPATH=/home/i/ines-wichert/.conda/envs/ines_pybrep:$PYTHONPATH
export LD_LIBRARY_PATH=/home/i/ines-wichert/.conda/envs/ines_pybrep/lib/python3.6/site_packages:$LD_LIBRARY_PATH

echo 'HOME =' $HOME
echo 'PATH =' $PATH
echo 'LD_LIBRARY_PATH =' $LD_LIBRARY_PATH
echo 'DISPLAY = ' $DISPLAY
unset DISPLAY # Turn of DISPLAY just in case

cd $SLURM_SUBMIT_DIR
echo 'WORKDIR =' $SLURM_SUBMIT_DIR

NENGS=$(($SLURM_NTASKS - 1))
echo 'NENGS = ' $NENGS

IPYTHONDIR=$SLURM_SUBMIT_DIR/$SLURM_JOB_ID.ipython #Create an ipython configuration directory
echo $IPYTHONDIR

ipython profile create --ipython-dir=$IPYTHONDIR
ipcontroller --ip='*' --ipython-dir=$IPYTHONDIR &

sleep 10

srun ipengine --ipython-dir=$IPYTHONDIR
