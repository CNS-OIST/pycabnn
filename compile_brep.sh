#!/bin/sh

sudo rm ./brep_deploy/brep_deploy
sudo rm ./brep_deploy/libchicken.so.7

echo -n 'Deleted old files, compiling new... '

#sudo csc -deploy -O3 -d0 -o ./brep_deploy -include-path ./brep_deploy brep.scm 
sudo csc -deploy -O3 -d0 -o ./brep_deploy -include-path ./brep_deploy brep.scm -o ./brep_exec


echo -n 'Compiled. Run now? (y/n): '
read ans

if echo "$ans" | grep -iq '^y' ;then 
	./brep_deploy/brep_deploy 
else 
	echo 'Okay, whatever. Bye!'
fi



