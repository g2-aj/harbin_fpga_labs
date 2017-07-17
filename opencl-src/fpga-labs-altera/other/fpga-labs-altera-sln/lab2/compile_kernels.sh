echo $1

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/altera/16.0/quartus/dspba/backend/linux64/
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/x86_64-linux-gnu

if [ "$1" = "hw" ]; then
	echo "Compiling kernels to generate hardware"
	aoc  conv_kernel.cl -o bin/conv_kernel.aoco -c -v --report --board de1soc_sharedonly
else
	echo "Compiling kernels for emulation"
	aoc -march=emulator conv_kernel.cl -o bin/conv_kernel.aocx -v --report --board s5_ref
fi
