#!/usr/bin/bash

#SBATCH --job-name=MLI_dend_gen
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=4g
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --output=gen_mli_dend_gen.out.log
#SBATCH --error=gen_mli_dend_gen.err.log

export NEURONHOME=/apps/unit/DeSchutterU/neuron/7.8.2
export PATH=$NEURONHOME/x86_64/bin:$PATH
export LD_LIBRARY_PATH=$NEURONHOME/lib/python:$NEURONHOME/x86_64/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$NEURONHOME/lib/python:$PYTHONPATH

nrngui -python generate_MLI_dend.py
