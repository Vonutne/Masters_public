This is the repository containing the code for the thesis: "Optimizing Convolutions for GPU execution in Futhark"
The code is largely split up in 2 parts, the CUDA code and the Futhark code.
The CUDA code contains makefiles for running and testing the code. 
The Futhark code contains makefiles for creating data sets. To benchmark the code, use futhark bench.

# Shared Memory Convolution
The shared memory version is located in ``Conv_cuda/shared_memory_conv.cu``. The main function runs a number of performance and validation tests. Compile the program with ``nvcc shared_memory_conv.cu -o shared_memory_conv`` and run it in your terminal.
# Conv_CUDA
The CUDA code contains makefiles for running and testing the code. If specific tiling or register parameters are wanted, they should be changed in the main.cu file.
To run the shared memory version, compile with nvcc and run, likewise to change parameters or input sizes, this should be done if file "?"
# Futhark code
The Futhark code contains makefiles for creating data sets. To benchmark the code, use futhark bench with desired data set
