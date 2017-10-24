#!/bin/sh

BREP=~/Desktop/LabRot_OIST/cerebellar_granular_network/brep/brep_deploy
INP=~/Desktop/LabRot_OIST/pybrep/input_files

sudo rm $BREP/brep_deploy
sudo rm $BREP/libchicken.so.7

echo -n 'Deleted old files, compiling new... '

sudo csc -deploy -O3 -d0 -o $BREP -include-path $BREP brep_commented.scm 
#sudo csc -deploy -O3 -d0 -o ./brep_deploy -include-path ./brep_deploy brep.scm -o ./brep_exec


echo -n 'Compiled. Run now? (y/n): '
read ans

if echo "$ans" | grep -iq '^y' ;then 
	#./brep_deploy/brep_deploy 
	mpirun $BREP/brep_deploy --gct-points=$INP/GCT_smallsmall.dat --config-file=$INP/Parameters.hoc --verbose
else 
	echo 'Okay, whatever. Bye!'
fi



