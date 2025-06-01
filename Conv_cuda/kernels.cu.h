#ifndef CONV_KERNELS
#define CONV_KERNELS


// block size must be the out put size of Input - radius +1, Very naive solution
template <class ElTp, int radius>
__global__ void convNaive (ElTp* input, ElTp* kernel, ElTp* output, int height, int widthIn, int widthOut, int nIn, int N_out){
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gidy = blockIdx.y * blockDim.y + threadIdx.y;
    int gidz = blockIdx.z * blockDim.z + threadIdx.z;
    if (gidx >= widthOut || gidy >= widthOut || gidz >= N_out){
        return;
    }
    ElTp sum = 0.0f;
    int r_size = 2*radius+1;
    for (int c_in = 0; c_in < nIn; c_in++) {
        for (int row = 0; row < r_size; row += 1) {
            for (int col = 0; col < r_size; col += 1 ) {
               int InCol = gidx + col;
               int InRow = gidy + row;
                sum += input[c_in * widthIn * widthIn + InRow * widthIn + InCol] *
                kernel[c_in * N_out * r_size * r_size + gidz * r_size * r_size + row * r_size + col];
            }
        }
    }
    output[gidz * widthOut * widthOut + gidy * widthOut + gidx] = sum;
}

/**
 * Used to copy slice X[iii: iii+T*R][kk:kk+Tk] from global to shared memory using Blen threads
 * isT == 0 => X is in row-major form, i.e., not transposed.
 * isT == 1 => X is in a delayed-transposed form.
 */


// This seems bad in form of coalesing, since threads does not access neigbouring elements, x should be the inner dimension



// row major version of 1D
template<class ElTp, int radius, int y_per_thread>
__global__ void conv1DTiledRM (ElTp* input, ElTp* kernel, ElTp* output, int height, int widthIn, int widthOut, int N_in, int N_out) {
    int gidx  = blockIdx.x * blockDim.x + threadIdx.x;
    int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * y_per_thread; // handle edge case where height % YPT != 0
    int c_out = blockIdx.z * blockDim.z + threadIdx.z;
    if (gidx >= widthOut || y0 >= widthOut || c_out >= N_out) {
        return;
    }
    ElTp tmp[y_per_thread];
    #pragma unroll
    for (int i = 0; i < y_per_thread; i++){
        tmp[i] = 0.0f;
    }
    int r_size = 2*radius+1;
    for (int c_in = 0; c_in < N_in; c_in++) {
        // might want some shared memory copying here
        #pragma unroll
        for (int iy = 0; iy < y_per_thread; iy ++) {
            int y = y0 + iy;
            if (y < widthOut){
                for (int i = 0; i < r_size; i ++) {
                    for (int j = 0 ; j < r_size; j++ ){
                        tmp[iy] += input[c_in * widthIn * widthIn + (y+i) * widthIn + (gidx+j)] *
                        kernel[c_in * N_out * r_size * r_size + c_out * r_size * r_size + i * r_size + j];
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int iy = 0; iy < y_per_thread; iy ++) {
        int y = y0 + iy;
        if (y < widthOut){
            output[c_out * widthOut * widthOut + y * widthOut + gidx] = tmp[iy];
        }
    }
}



// Row major version of 2d
template<class ElTp, int radius, int Ry,int Rz>
__global__ void conv2DTiledRM (ElTp* input, ElTp* kernel, ElTp* output, int height, int widthIn, int widthOut, int N_in, int N_out) {
    int gidx  = blockIdx.x * blockDim.x + threadIdx.x;
    int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * Ry;
    int c_out0 = blockIdx.z * blockDim.z;
    if (blockDim.z != 1) {
        c_out0 += threadIdx.z;
    }
    c_out0 = c_out0 * Rz;


    ElTp tmp[Ry][Rz];
    #pragma unroll
    for (int i =0; i< Ry*Rz; i++){
        #pragma unroll
        for (int j =0; j<Rz; j++)
            tmp[i][j] = 0.0;
    }
    if (gidx >= widthOut || y0 >= widthOut || c_out0 >= N_out) {
        return;
    }
    int r_size = 2*radius+1;
    for (int c_in = 0; c_in < N_in; c_in++) {
        // might want some shared memory copying here
        // use Lmad way of copying with padding, so no branches in the code
        #pragma unroll
        for (int iy = 0; iy < Ry; iy ++) {
            int y = y0 + iy;
            #pragma unroll
            for (int ic = 0; ic < Rz; ic++) {
                int c_out =  c_out0 + ic;
                #pragma unroll
                for (int i = 0; i < r_size; i ++) {
                    #pragma unroll
                    for (int j = 0 ; j < r_size; j++ ){
                        if (y < widthOut && c_out < N_out){
                            tmp[iy][ic] += input[c_in * widthIn * widthIn + (y+i) * widthIn + (gidx+j)] *
                            kernel[c_in * N_out * r_size * r_size + c_out * r_size * r_size + i * r_size + j];
                        }
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int iy = 0; iy < Ry; iy ++) {
        int y = y0 + iy;
        #pragma unroll
        for (int ic = 0; ic < Rz; ic++) {
            int c_out = c_out0 + ic;
            if (c_out < N_out && y < widthOut){
                output[c_out * widthOut * widthOut + y * widthOut + gidx] = tmp[iy][ic];
            }
        }
    }
}


template<class ElTp, int radius, int Ry,int Rz>
__global__ void conv2DTiledBL (ElTp* input, ElTp* kernel, ElTp* output, int height, int widthIn, int widthOut, int N_in, int N_out) {
    int gidx  = blockIdx.x * blockDim.x + threadIdx.x;
    int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * Ry;
    int c_out0 = blockIdx.z * blockDim.z;
    if (blockDim.z != 1) {
        c_out0 += threadIdx.z;
    }
    c_out0 = c_out0 * Rz;


    ElTp tmp[Ry][Rz];
    #pragma unroll
    for (int i =0; i< Ry*Rz; i++){
        #pragma unroll
        for (int j =0; j<Rz; j++)
            tmp[i][j] = 0.0;
    }
    int r_size = 2*radius+1;
    for (int c_in = 0; c_in < N_in; c_in++) {
        // might want some shared memory copying here
        // use Lmad way of copying with padding, so no branches in the code
        #pragma unroll
        for (int iy = 0; iy < Ry; iy ++) {
            int y = y0 + iy;
            #pragma unroll
            for (int ic = 0; ic < Rz; ic++) {
                int c_out =  c_out0 + ic;
                #pragma unroll
                for (int i = 0; i < r_size; i ++) {
                    #pragma unroll
                    for (int j = 0 ; j < r_size; j++ ){
                            tmp[iy][ic] += input[c_in * widthIn * widthIn + (y+i) * widthIn + (gidx+j)] *
                            kernel[c_in * N_out * r_size * r_size + c_out * r_size * r_size + i * r_size + j];
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int iy = 0; iy < Ry; iy ++) {
        int y = y0 + iy;
        #pragma unroll
        for (int ic = 0; ic < Rz; ic++) {
            int c_out = c_out0 + ic;
            if (c_out < N_out && y < widthOut && gidx < widthOut){
                output[c_out * widthOut * widthOut + y * widthOut + gidx] = tmp[iy][ic];
            }
        }
    }
}

#endif