#include <device_launch_parameters.h>
#include "helper.h"
#include "kernels.cu.h"

#define GPU_RUNS 5
#define ERR      0.000001

template<class T, int Tx, int Ty, int Tz, int radius>
__host__ void runNaive(T* d_in, T* d_ker, T*d_out, T* h_out, int n_in, int in_dim,
                 int n_out){
    int out_dim = (in_dim - 2 *radius);
    unsigned long long mem_size_out = out_dim * out_dim  * n_out * sizeof(T);
    cudaMemset(d_out, 0, mem_size_out);

    //set up parameters
    int dimy = (out_dim + Ty -1) / Ty;
    int dimx = (out_dim + Tx -1) / Tx;
    int dimz = (n_out + Tz -1) / Tz;

    dim3 block(Tx,Ty,Tz);
    dim3 grid (dimx,dimy,dimz);

    // dry runs 
    for (int i = 0; i < 3; i++){
        convNaive<T,radius> <<< grid, block >>>(d_in,d_ker,d_out,in_dim,in_dim,out_dim,n_in,n_out);
    }
    cudaDeviceSynchronize();
    gpuAssert(cudaPeekAtLastError());

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for (int i = 0; i < GPU_RUNS; i++){
        convNaive<T,radius> <<< grid, block >>>(d_in,d_ker,d_out,in_dim,in_dim,out_dim,n_in,n_out);
    }
    cudaDeviceSynchronize();
    gpuAssert(cudaPeekAtLastError());

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;
    
    double ker_size = radius * 2 +1;
    double flopsPerConv = 2.0 * n_out * out_dim * out_dim * ker_size * ker_size * n_in + n_out * out_dim * out_dim;
    double gigaFlops = (flopsPerConv * 1.0e-3f) / elapsed;

    printf("GPU Naive Conv version runs in %lu microsecs, Gflops/sec %.2f\n"
            , elapsed, gigaFlops);

    cudaMemcpy(h_out,d_out, mem_size_out,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

template<class T, int Tx, int Ty, int Ry, int Tz, int radius>
__host__ void run1DConv(T* d_in, T* d_ker, T*d_out, T* h_out, int n_in, int in_dim, int n_out){
    int out_dim = (in_dim - 2 *radius);
    unsigned long long size_out = out_dim * out_dim * n_out;
    unsigned long long mem_size_out = size_out * sizeof(T);
    T* h_check = (T*) malloc(mem_size_out);
    //Setup exection parameters
    int dimy = ceil(((float) out_dim/(Ty*Ry)));
    int dimx = (out_dim + Tx -1) / Tx;
    int dimz = (n_out + Tz -1) / Tz;
    dim3 block(Tx,Ty,Tz);
    dim3 grid (dimx, dimy, dimz);


    //test run
    for(int i = 0; i < 3; i++) {
        conv1DTiledRM<T,radius,Ry> <<< grid,block >>>(d_in,d_ker,d_out,in_dim,in_dim,out_dim,n_in,n_out);
    }
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError());
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    cudaMemset(d_out, 0, mem_size_out);
    gettimeofday(&t_start, NULL);
    for(int i=0; i < GPU_RUNS; i++){
        conv1DTiledRM<T,radius,Ry> <<< grid,block >>>(d_in,d_ker,d_out,in_dim,in_dim,out_dim,n_in,n_out);
    }
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError());

 
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;

    double ker_size = radius * 2 +1;
    double flopsPerConv = 2.0 * n_out * out_dim * out_dim * ker_size * ker_size * n_in + n_out * out_dim * out_dim;
    double gigaFlops = (flopsPerConv * 1.0e-3f) / elapsed;

    printf("GPU 1D Conv version runs in %lu microsecs, Gflops/sec %.2f\n"
            , elapsed, gigaFlops);

    // check validity
    cudaMemcpy(h_check, d_out, mem_size_out, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    validate<T>(h_out, h_check, size_out,ERR);
    //printZero<T>(h_out, size_out);
    //printZero<T>(h_check,size_out); 
    free(h_check);
}

template<class T, int Tx, int Ty, int Ry, int Tz, int Rz, int radius>
__host__ void run2DConv(T* d_in, T* d_ker, T*d_out, T* h_out, int n_in, int in_dim, int n_out){
    int out_dim = (in_dim - 2 *radius);
    unsigned long long size_out = out_dim * out_dim  * n_out;
    unsigned long long mem_size_out = size_out * sizeof(T);

    T* h_check = (T*) malloc(mem_size_out);

    //Setup exection parameters
    int dimy = ceil(((float) out_dim/(Ty*Ry)));
    int dimx = (out_dim + Tx -1) / Tx;
    int dimz = ceil(((float) n_out/(Tz*Rz)));
    dim3 block(Tx,Ty,Tz);
    dim3 grid (dimx, dimy, dimz);


    //test run
    for(int i = 0; i < 3; i++) {
        conv2DTiledRM<T,radius,Ry,Rz> <<< grid,block >>>(d_in,d_ker,d_out,in_dim,in_dim,out_dim,n_in,n_out);
    }
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError());
    cudaMemset(d_out, 0, mem_size_out);
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);
    for(int i=0; i < GPU_RUNS; i++){
        conv2DTiledRM<T,radius,Ry,Rz> <<< grid,block >>>(d_in,d_ker,d_out,in_dim,in_dim,out_dim,n_in,n_out);
    }
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError());

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;

    double ker_size = radius * 2 +1;
    double flopsPerConv = 2.0 * n_out * out_dim * out_dim * ker_size * ker_size * n_in;
    double gigaFlops = (flopsPerConv * 1.0e-3f) / elapsed;

    printf("GPU 2D Conv version runs in %lu microsecs, Gflops/sec %.2f\n"
            , elapsed, gigaFlops);
        // check validity
    cudaMemcpy(h_check, d_out, mem_size_out, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    //validate<T>(h_out, h_check, size_out,ERR);
    free(h_check);
}


template<class T, int Tx, int Ty, int Ry, int Tz, int Rz, int radius>
__host__ void run2DConvShm(T* d_in, T* d_ker, T*d_out, T* h_out, int n_in, int in_dim, int n_out){
    int out_dim = (in_dim - 2 *radius);
    unsigned long long size_out = out_dim * out_dim  * n_out;
    unsigned long long mem_size_out = size_out * sizeof(T);

    T* h_check = (T*) malloc(mem_size_out);

    //Setup exection parameters
    int dimy = ceil(((float) out_dim/(Ty*Ry)));
    int dimx = (out_dim + Tx -1) / Tx;
    int dimz = ceil(((float) out_dim/(Tz*Rz)));
    dim3 block(Tx,Ty,Tz);
    dim3 grid (dimx, dimy, dimz);

    const size_t shmem_size = (Ty * Ry + 2 *radius) * (Tx + 2 * radius) * sizeof(T);

    //test run
    for(int i = 0; i < 3; i++) {
        conv2DTiledShm<T,radius,Ty,Ry,Tx,Tz,Rz> <<< grid,block, shmem_size >>>(d_in,d_ker,d_out,in_dim,in_dim,out_dim,n_in,n_out);
    }
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError());
    cudaMemset(d_out, 0, mem_size_out);
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);
    for(int i=0; i < GPU_RUNS; i++){
        conv2DTiledShm<T,radius,Ty,Ry,Tx,Tz,Rz> <<< grid,block, shmem_size >>>(d_in,d_ker,d_out,in_dim,in_dim,out_dim,n_in,n_out);
    }
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError());

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;

    double ker_size = radius * 2 +1;
    double flopsPerConv = 2.0 * n_out * out_dim * out_dim * ker_size * ker_size * n_in;
    double gigaFlops = (flopsPerConv * 1.0e-3f) / elapsed;

    printf("GPU 2D Shmm version runs in %lu microsecs, Gflops/sec %.2f\n"
            , elapsed, gigaFlops);
        // check validity
    cudaMemcpy(h_check, d_out, mem_size_out, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    //validate<T>(h_out, h_check, size_out,ERR);
   
   
   
    free(h_check);
}


template<class T, int Tx, int Ty, int Ry, int Tz, int Rz, int radius>
__host__ void run2DConvBL(T* d_in, T* d_ker, T*d_out, T* h_out, int n_in, int in_dim, int n_out){
    int out_dim = (in_dim - 2 *radius);
    unsigned long long size_out = out_dim * out_dim  * n_out;
    unsigned long long mem_size_out = size_out * sizeof(T);

    T* h_check = (T*) malloc(mem_size_out);

    //Setup exection parameters
    int dimy = ceil(((float) out_dim/(Ty*Ry)));
    int dimx = (out_dim + Tx -1) / Tx;
    int dimz = ceil(((float) n_out/(Tz*Rz)));
    dim3 block(Tx,Ty,Tz);
    dim3 grid (dimx, dimy, dimz);


    //test run
    for(int i = 0; i < 3; i++) {
        conv2DTiledRM<T,radius,Ry,Rz> <<< grid,block >>>(d_in,d_ker,d_out,in_dim,in_dim,out_dim,n_in,n_out);
    }
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError());
    cudaMemset(d_out, 0, mem_size_out);
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);
    for(int i=0; i < GPU_RUNS; i++){
        conv2DTiledBL<T,radius,Ry,Rz> <<< grid,block >>>(d_in,d_ker,d_out,in_dim,in_dim,out_dim,n_in,n_out);
    }
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError());

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;

    double ker_size = radius * 2 +1;
    double flopsPerConv = 2.0 * n_out * out_dim * out_dim * ker_size * ker_size * n_in;
    double gigaFlops = (flopsPerConv * 1.0e-3f) / elapsed;

    printf("GPU 2D Branchless Conv version runs in %lu microsecs, Gflops/sec %.2f\n"
            , elapsed, gigaFlops);
        // check validity
    cudaMemcpy(h_check, d_out, mem_size_out, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    //validate<T>(h_out, h_check, size_out,ERR);
    free(h_check);
}



template<class T, int Tx, int Ty, int Ry, int Tz, int Rz, int radius>
__host__ void runVersions (int in_dim, int n_in, int n_out) {

    srand(4242);

    //1.1 allocate memory for input, output and kernel channels. Right now assuming floats, but might want
    unsigned long long in_size = n_in * in_dim * in_dim;
    unsigned long long in_size_mem = in_size * sizeof(T) + (in_dim * in_dim) * sizeof(T); // extra padding for branchless
    T* h_in = (T*) malloc(in_size_mem);

    int ker_dim = 2*radius+1;
    unsigned long long radius_size = n_in * n_out * ker_dim * ker_dim;
    unsigned long long radius_size_mem = radius_size * sizeof(T);
    T* h_ker = (T*) malloc(radius_size_mem);
    // 1.2 formula for the resulting output channel
    unsigned long long out_n = in_dim - 2 * radius;
    unsigned long long out_size = out_n * out_n * n_out;
    unsigned long long out_size_mem = out_size * sizeof(T);
    T* h_out = (T*) malloc(out_size_mem);
    // 2. Initialize memory using the helper function
    randomInit<T>(h_in, in_size);
    randomInit<T>(h_ker, radius_size);

    // 3. allocate device memory

    T* d_in;
    T* d_ker;
    T* d_out;
    cudaMalloc((void**) &d_in, in_size_mem);
    cudaMalloc((void**) &d_ker, radius_size_mem);
    cudaMalloc((void **) &d_out, out_size_mem);
    //printf("in pointer: %p, ker pointer: %p, out pointer %p, \n",d_in,d_ker,d_out);
    // 4. copy host to device
    cudaMemcpy(d_in, h_in, in_size_mem, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ker, h_ker, radius_size_mem, cudaMemcpyHostToDevice);
    printf("In_dim : %d, out dim %d, ker dim %d, Nin %d, Nout %d \n", in_dim, out_n, ker_dim,n_in,n_out);
    // run some code
    
    // run naive
    #if 0
    
    // run 1D
    run1DConv<T,Tx,Ty,Ry,Tz,radius> (d_in,d_ker,d_out,h_out,n_in,in_dim,n_out);
    // run 2D
    run2DConv<T,Tx,Ty,Ry,Tz,Rz,radius> (d_in,d_ker,d_out,h_out,n_in,in_dim,n_out);
    #endif
    #if 1
    runNaive<T,Tx,Ty,Tz,radius> (d_in, d_ker,d_out,h_out,n_in,in_dim,n_out);
    run2DConv<T,Tx,Ty,Ry,Tz,Rz,radius> (d_in,d_ker,d_out,h_out,n_in,in_dim,n_out);
    run2DConvBL<T,Tx,Ty,Ry,Tz,Rz,radius> (d_in,d_ker,d_out,h_out,n_in,in_dim,n_out);
    #endif
    // run shared
    //run2DConvShm<T,Tx, Ty,Ry,Tz,Rz,radius> (d_in, d_ker, d_out, h_out, n_in, in_dim, n_out);

    //clean up
    printf("Sizes are: (Indim %d, Outdim %d, radius %d, nIn %d, nOut %d, Tx :%d, Ty %d, Ry %d, Tz %d, R< %d) \n",
                        in_dim, out_n, radius, n_in, n_out,Tx,Ty,Ry,Tz, Rz);
    free(h_in);
    free(h_ker);
    free (h_out);
    cudaFree(d_in);
    cudaFree(d_ker);
    cudaFree(d_out);

}


int main (int argc, char * argv[]) {
    if (argc != 4) {
        printf("Usage: %s in_dim N_out N_IN width ker-size\n", argv[0]);
        return 0;
    }
    // right now not handling strides, only assuming 1 in -> 1 out,  and that all
    // padding is already handled
    const int IN_DIM = atoi(argv[1]);
    const int N_OUT = atoi(argv[2]);
    const int N_IN = atoi(argv[3]);

    cudaSetDevice(0);

    #if 0
    //printf("we get here ? \n");
    runVersions<float, 16,16,3,2,3,1>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 16,16,3,2,3,1>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 32,8,8,4,1,1>(IN_DIM,N_IN,N_OUT);
    runVersions<float,256,1,8,1,8,1>(IN_DIM,N_IN,N_OUT);
    runVersions<float,256,1,8,1,8,2>(IN_DIM,N_IN,N_OUT);
    runVersions<float,256,1,8,1,8,3>(IN_DIM,N_IN,N_OUT);
    runVersions<float,256,1,8,1,8,1>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 32,8,8,4,1,1>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 32,8,2,4,2,3>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 32,8,2,4,4,5>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 64,4,4,2,8,5>(IN_DIM,N_IN,N_OUT);
    runVersions<float,256,1,8,1,8,1>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 32,4,8,2,4,1>(IN_DIM,N_IN,N_OUT);
    //printf("Running double versions \n");
    //runVersions<double, 16,8,8,2,1,1>(IN_DIM,N_IN,N_OUT);
    //runVersions<double, 16,8,8,2,1,3>(IN_DIM,N_IN,N_OUT);

    //best versions of float4 version
    runVersions<float, 8,2,32,2,2,2>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 8,2,16,2,2,3>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 8,2,16,6,4,4>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 8,2,14,2,1,5>(IN_DIM,N_IN,N_OUT);
    //template<class T, int Tx, int Ty, int Ry, int Tz, int Rz, int radius>
    //versions for evaluation:
    #endif
    #if 1
    runVersions<float, 32, 2,13,2,4,1>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 32, 2,10,2,4,2>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 32, 2,16,2,2,3>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 32, 2,8,2,4,4>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 32, 2,6,2,4,5>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 32, 2,5,2,2,6>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 32, 2,5,2,2,7>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 32, 2,8,2,2,8>(IN_DIM,N_IN,N_OUT);
    runVersions<float,256,1,8,1,8,1>(IN_DIM,N_IN,N_OUT);
    #endif
    // versions for 1x1 eval
    #if 0
    runVersions<float, 32, 4,13,1,1,1>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 32, 4,10,1,1,2>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 32, 4,21,1,1,3>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 32, 4,13,1, 1,4>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 32, 4,11,1, 1,5>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 32, 4,10,1, 1,6>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 32, 4,5, 1, 1,7>(IN_DIM,N_IN,N_OUT);
    runVersions<float, 32, 4,8, 1, 1,8>(IN_DIM,N_IN,N_OUT);
    #endif


    //runVersions<float, 8,32,14,2,8,1>(IN_DIM,N_IN,N_OUT);
    
    

}