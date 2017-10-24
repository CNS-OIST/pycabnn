#!/bin/sh

BREP=~/Desktop/LabRot_OIST/cerebellar_granular_network/brep/brep_deploy
INP=~/Desktop/LabRot_OIST/pybrep/input_files
RNG_SEEDS="73,79,83,89,97,101,103,107,109,113"

sudo rm $BREP/brep_deploy
sudo rm $BREP/libchicken.so.7

echo -n 'Deleted old files, compiling new... '

sudo csc -deploy -O3 -d0 -o $BREP -include-path $BREP brep_commented.scm 
#sudo csc -deploy -O3 -d0 -o ./brep_deploy -include-path ./brep_deploy brep.scm -o ./brep_exec


echo -n 'Compiled. Run now? (y/n): '
read ans

if echo "$ans" | grep -iq '^y' ;then 
	#./brep_deploy/brep_deploy 
	mpirun $BREP/brep_deploy --rng-seeds="$RNG_SEEDS" --gct-points=$INP/GCT_smallsmall.dat --config-file=$INP/Parameters.hoc --verbose  \
	> terminal_output.txt
else 
	echo 'Okay, whatever. Bye!'
fi



