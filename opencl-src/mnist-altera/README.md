# Description
This project is the FPGA implementation of MNIST digit classification. The project uses Altera OpenCL for high-level synthesis and hecen requires the AOCL SDK  and Quartus tools for compilation and execution.

# To run
- Setup Altera specific environment variables.
- The kernels need to compiled before compiling host application. The kernel compilation can be done for emulation mode or hardware mode(takes hours to compile).
  * To compile for emulation mode, run _./compile_kernels.sh_
  * To compile full hardware btistream, run _./compile_kernels.sh hw_
- The above compilation should produce _*.aocx_ file inside _./bin_ directory.
- To compile and run the application for emulation, do "make run_emu_sample" (this compiles the host code and runs the mnist application with a sample image). To run a set of images do "make run_emu_sample"

  * To compile the host application for the ARM processor on the DE1-SoC, do "make hw"

You need to setup the Altera FPGA board before compiling. Refer to [Altera getting started guide](https://www.altera.com/content/dam/altera-www/global/en_US/pdfs/literature/hb/opencl-sdk/aocl_getting_started.pdf) or [AOCL programming guide](https://www.altera.com/en_US/pdfs/literature/hb/opencl-sdk/aocl_programming_guide.pdf) to setup the board and the environments required for compilation.
