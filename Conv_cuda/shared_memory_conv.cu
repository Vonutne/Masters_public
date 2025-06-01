
// ReSharper disable CommentTypo
// ReSharper disable GrammarMistakeInComment
#include <cstdlib>
#include <cstring>

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef unsigned long long u64;
typedef long long i64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
typedef float f32;
typedef double f64;

#define gpuErrchk(ans) (gpuAssert((ans), __FILE__, __LINE__))
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__device__ inline int device_index_arr_of_arr(const int pitch_row, const int height, const int z, const int y, const int x) {
	return pitch_row * height * z + pitch_row * y + x;
}

__forceinline__ int index_arr_of_arr(const int pitch_row, const int height, const int z, const int y, const int x) {
	return pitch_row * height * z + pitch_row * y + x;
}

__forceinline__ int index_filters(const int cout_per_block, const int in_channel_pitch, const int block_pitch, const int filter_width, const int in_channel, const int out_channel, const int y, const int x) {
	int block = (out_channel / cout_per_block);
	int out_within_block = (out_channel % cout_per_block);
	return in_channel_pitch * in_channel + block_pitch * block + filter_width * filter_width * out_within_block + filter_width * y + x;
}

__device__ int device_index_filters(const int cout_per_block, const int in_channel_pitch, const int block_pitch, const int filter_width, const int in_channel, const int out_channel, const int y, const int x) {
	int block = (out_channel / cout_per_block);
	int out_within_block = (out_channel % cout_per_block);
	return in_channel_pitch * in_channel + block_pitch * block + filter_width * filter_width * out_within_block + filter_width * y + x;
}

#define ROUND_UP_TO_MULTIPLE_OF(numToRound, multiple) ((((numToRound) + (multiple) - 1) / (multiple)) * (multiple))

template <class T, class LoadType, int FilterRadius, int Tx, int Ty, int Tz, int Rz, int Ry>
__global__ void convolution_shared_with_reduction(
	T* __restrict__ input,
	T* __restrict__ filters,
	T* __restrict__ out,
	int in_num_channels,
	int in_height,
	int in_row_pitch,
	int filters_in_channel_pitch,
	int filters_block_pitch,
	int out_num_channels,
	int out_width,
	int out_height,
	int out_row_pitch
) {
	constexpr int CoutPerBlock = Tz * Rz;
	constexpr int YPerBlock = Ty * Ry;
	constexpr int ElementsPerLoadType = sizeof(LoadType) / sizeof(T);

	constexpr int FilterWidth = 2 * FilterRadius + 1;
	constexpr int FiltersSharedSize = CoutPerBlock * FilterWidth * FilterWidth;
	constexpr int InSharedWidth = Tx + 2 * FilterRadius;
	constexpr int InSharedPitch = ROUND_UP_TO_MULTIPLE_OF(InSharedWidth, ElementsPerLoadType);
	constexpr int InSharedHeight = YPerBlock + 2 * FilterRadius;
	constexpr int InSharedSize = InSharedPitch * InSharedHeight;
	constexpr int NumThreads = Tx * Ty * Tz;

	constexpr int FiltersNumLoadType = (FiltersSharedSize + ElementsPerLoadType - 1) / ElementsPerLoadType;

	constexpr int InSharedLoadTypePerRow = (InSharedWidth + ElementsPerLoadType - 1) / ElementsPerLoadType;
	constexpr int NumLoadTypes = InSharedLoadTypePerRow * InSharedHeight;

	volatile extern __shared__ int shared_convolution_shared_load_LoadType[];
	T* in_shared = (T*)shared_convolution_shared_load_LoadType;
	T* filters_shared = &in_shared[ROUND_UP_TO_MULTIPLE_OF(InSharedSize, NumThreads * ElementsPerLoadType)];

	const int block_offset_x = blockIdx.x * Tx;
	const int block_offset_y = blockIdx.y * YPerBlock;
	const int filters_block_offset = blockIdx.z * filters_block_pitch;

	const int thread_id = threadIdx.x + threadIdx.y * Tx + threadIdx.z * Tx * Ty; // [0 : num_threads_per_block)

	T tmp[Ry][Rz];
#pragma unroll
	for (int i = 0; i < Ry; i++) {
#pragma unroll
		for (int j = 0; j < Rz; j++) {
			tmp[i][j] = 0.0;
		}
	}

	for (int in_channel = 0; in_channel < in_num_channels; in_channel++) {
#pragma unroll
		for (int load_start = 0; load_start < NumLoadTypes; load_start += NumThreads) {
			const int id = load_start + thread_id;
			const int block_y = id / InSharedLoadTypePerRow;
			const int block_x = (id % InSharedLoadTypePerRow) * ElementsPerLoadType;
			((LoadType*)in_shared)[id] =
				*(LoadType*)&input[device_index_arr_of_arr(in_row_pitch, in_height, in_channel, block_offset_y + block_y, block_offset_x + block_x)];
		}
#pragma unroll
		for (int load_start = 0; load_start < FiltersNumLoadType; load_start += NumThreads) {
			const int id = load_start + thread_id;
			((LoadType*)filters_shared)[id] =
				*(LoadType*)&filters[filters_in_channel_pitch * in_channel + filters_block_offset + id * ElementsPerLoadType];
		}
		__syncthreads();

		// --------------- Compute shit ---------------
#pragma unroll
		for (int loop_y = 0; loop_y < Ry; loop_y++) {
			const int y = threadIdx.y * Ry + loop_y;
#pragma unroll
			for (int thread_cout = 0; thread_cout < Rz; thread_cout++) {
#pragma unroll
				for (int i = 0; i < FilterWidth; i++) {
#pragma unroll
					for (int j = 0; j < FilterWidth; j++) {
						tmp[loop_y][thread_cout] +=
							in_shared[InSharedPitch * (y + i) + threadIdx.x + j] *
							filters_shared[FilterWidth * FilterWidth * (Rz * threadIdx.z + thread_cout) + FilterWidth * i + j];
					}
				}
			}
		}
		__syncthreads();
	}

#pragma unroll
	for (int loop_y = 0; loop_y < Ry; loop_y++) {
		const int y = block_offset_y + Ry * threadIdx.y + loop_y;
#pragma unroll
		for (int thread_cout = 0; thread_cout < Rz; thread_cout++) {
			const int cout = CoutPerBlock * blockIdx.z + Rz * threadIdx.z + thread_cout;
			if (cout < out_num_channels && y < out_height && block_offset_x + threadIdx.x < out_width) {
				out[device_index_arr_of_arr(out_row_pitch, out_height, cout, y, block_offset_x + threadIdx.x)] = tmp[loop_y][thread_cout];
			}
		}
	}
}

// This kernel pads both the input and output array such that the output array can be re-used as the input
// array in the next iteration. 
template <class T, class LoadType, int FilterRadius, int Tx, int Ty, int Tz, int Rz, int Ry>
__global__ void convolution_shared(
	T* __restrict__ input,
	T* __restrict__ filters,
	T* __restrict__ out,
	int in_num_channels,
	int height,
	int width,
	int row_pitch,
	int filters_in_channel_pitch,
	int filters_block_pitch,
	int out_num_channels
) {
	constexpr int CoutPerBlock = Tz * Rz;
	constexpr int YPerBlock = Ty * Ry;
	constexpr int ElementsPerLoadType = sizeof(LoadType) / sizeof(T);

	constexpr int FilterWidth = 2 * FilterRadius + 1;
	constexpr int FiltersSharedSize = CoutPerBlock * FilterWidth * FilterWidth;
	constexpr int InSharedWidth = Tx + 2 * FilterRadius;
	constexpr int InSharedPitch = ROUND_UP_TO_MULTIPLE_OF(InSharedWidth, ElementsPerLoadType);
	constexpr int InSharedHeight = YPerBlock + 2 * FilterRadius;
	constexpr int InSharedSize = InSharedPitch * InSharedHeight;
	constexpr int NumThreads = Tx * Ty * Tz;

	constexpr int FiltersNumLoadType = (FiltersSharedSize + ElementsPerLoadType - 1) / ElementsPerLoadType;

	constexpr int InSharedLoadTypePerRow = (InSharedWidth + ElementsPerLoadType - 1) / ElementsPerLoadType;
	constexpr int NumLoadTypes = InSharedLoadTypePerRow * InSharedHeight;

	volatile extern __shared__ int shared_convolution_shared_load_LoadType[];
	T* in_shared = (T*)shared_convolution_shared_load_LoadType;
	T* filters_shared = &in_shared[ROUND_UP_TO_MULTIPLE_OF(InSharedSize, NumThreads * ElementsPerLoadType)];

	const int block_offset_x = blockIdx.x * Tx;
	const int block_offset_y = blockIdx.y * YPerBlock;
	const int filters_block_offset = blockIdx.z * filters_block_pitch;

	const int thread_id = threadIdx.x + threadIdx.y * Tx + threadIdx.z * Tx * Ty; // [0 : num_threads_per_block)

	T tmp[Ry][Rz];
#pragma unroll
	for (int i = 0; i < Ry; i++) {
#pragma unroll
		for (int j = 0; j < Rz; j++) {
			tmp[i][j] = 0.0;
		}
	}

	for (int in_channel = 0; in_channel < in_num_channels; in_channel++) {
#pragma unroll
		for (int load_start = 0; load_start < NumLoadTypes; load_start += NumThreads) {
			const int id = load_start + thread_id;
			const int block_y = id / InSharedLoadTypePerRow;
			const int block_x = (id % InSharedLoadTypePerRow) * ElementsPerLoadType;
			((LoadType*)in_shared)[id] =
				*(LoadType*)&input[device_index_arr_of_arr(row_pitch, height, in_channel, block_offset_y + block_y, block_offset_x + block_x)];
		}
#pragma unroll
		for (int load_start = 0; load_start < FiltersNumLoadType; load_start += NumThreads) {
			const int id = load_start + thread_id;
			((LoadType*)filters_shared)[id] = *(LoadType*)&filters[filters_in_channel_pitch * in_channel +
				filters_block_offset + id * ElementsPerLoadType];
		}
		__syncthreads();

		// --------------- Compute shit ---------------
#pragma unroll
		for (int loop_y = 0; loop_y < Ry; loop_y++) {
			const int y = threadIdx.y * Ry + loop_y;
#pragma unroll
			for (int thread_cout = 0; thread_cout < Rz; thread_cout++) {
#pragma unroll
				for (int i = 0; i < FilterWidth; i++) {
#pragma unroll
					for (int j = 0; j < FilterWidth; j++) {
						tmp[loop_y][thread_cout] +=
							in_shared[InSharedPitch * (y + i) + threadIdx.x + j] *
							filters_shared[FilterWidth * FilterWidth * (Rz * threadIdx.z + thread_cout) + FilterWidth * i + j];
					}
				}
			}
		}
		__syncthreads();
	}

#pragma unroll
	for (int loop_y = 0; loop_y < Ry; loop_y++) {
		const int y = block_offset_y + Ry * threadIdx.y + loop_y + FilterRadius;
#pragma unroll
		for (int thread_cout = 0; thread_cout < Rz; thread_cout++) {
			const int cout = CoutPerBlock * blockIdx.z + Rz * threadIdx.z + thread_cout;
			if (cout < out_num_channels && y < height && block_offset_x + threadIdx.x + FilterRadius < width) {
				out[device_index_arr_of_arr(row_pitch, height, cout, y, block_offset_x + threadIdx.x + FilterRadius)] = tmp[loop_y][thread_cout];
			}
		}
	}
}

int gpuAssert(cudaError_t code) {
	if (code != cudaSuccess) {
		printf("GPU Error: %s\n", cudaGetErrorString(code));
		return 0;
	}
	return 1;
}

template<class T>
__global__ void device_init_arr(T* arr, T value, u64 size) {
	u64 gId = threadIdx.x + blockIdx.x * blockDim.x;

	if (gId < size) {
		arr[gId] = value;
	}
}

f64 flops_from_ms(int out_width, int out_height, int filter_radius, int in_num_channels, int out_num_channels, f64 ms) {
	return (f64)out_width * (f64)out_height * (f64)in_num_channels * (f64)out_num_channels * (2.0 * (f64)filter_radius + 1.0) * (2.0 * (f64)filter_radius + 1.0) * 2 / ms;
}

void print_FLOPS(const char* name, cudaError_t init_error, cudaError_t kernel_error, FILE* fptr, int FilterRadius, int Tx, int Ty, int Tz, int Rz, int Ry, f64 flops) {
	if (gpuAssert(init_error)) {
		if (gpuAssert(kernel_error)) {
			fprintf(fptr, "%s execution succeeded (FilterRadius=%d, Tx= %d, Ty= %d, Tz= %d, Rz= %d, Ry= %d): %e FLOPS\n", name, FilterRadius, Tx, Ty, Tz, Rz, Ry, flops);
			printf("%s execution succeeded (FilterRadius=%d, Tx= %d, Ty= %d, Tz= %d, Rz= %d, Ry= %d): %e FLOPS\n", name, FilterRadius, Tx, Ty, Tz, Rz, Ry, flops);
		}
		else {
			fprintf(fptr, "%s execution failed (FilterRadius=%d, Tx= %d, Ty= %d, Tz= %d, Rz= %d, Ry= %d): %s\n", name, FilterRadius, Tx, Ty, Tz, Rz, Ry, cudaGetErrorString(kernel_error));
			printf("%s execution failed (FilterRadius=%d, Tx= %d, Ty= %d, Tz= %d, Rz= %d, Ry= %d): %s\n", name, FilterRadius, Tx, Ty, Tz, Rz, Ry, cudaGetErrorString(kernel_error));
		}
	}
	else {
		fprintf(fptr, "Init failed (FilterRadius=%d, Tx= %d, Ty= %d, Tz= %d, Rz= %d, Ry= %d): %s\n", FilterRadius, Tx, Ty, Tz, Rz, Ry, cudaGetErrorString(init_error));
		printf("Init failed (FilterRadius=%d, Tx= %d, Ty= %d, Tz= %d, Rz= %d, Ry= %d): %s\n", FilterRadius, Tx, Ty, Tz, Rz, Ry, cudaGetErrorString(init_error));
	}
}

typedef struct {
	dim3 grid_size;
	dim3 block_size;
	u64 shared_memory_size;
	u64 input_size;
	u64 filters_size;
	u64 out_size;
	int out_width;
	int out_height;
	int in_row_pitch;
	int filters_in_channel_pitch;
	int filters_block_pitch;
	int out_row_pitch;
} convolution_with_reduction_sizes_t;

typedef struct {
	dim3 grid_size;
	dim3 block_size;
	u64 shared_memory_size;
	u64 input_size;
	u64 filters_size;
	u64 out_size;
	int padded_width;
	int padded_height;
	int row_pitch;
	int filters_in_channel_pitch;
	int filters_block_pitch;
} convolution_sizes_t;

template <class T, class LoadType, int FilterRadius, int Tx, int Ty, int Tz, int Rz, int Ry>
convolution_with_reduction_sizes_t get_convolution_with_reduction_sizes(
	int in_num_channels,
	int in_width,
	int in_height,
	int out_num_channels
) {
	constexpr int ElementsPerLoadType = sizeof(LoadType) / sizeof(T);
	constexpr int FilterWidth = FilterRadius * 2 + 1;
	constexpr int CoutPerBlock = Tz * Rz;
	constexpr int YPerBlock = Ry * Ty;
	constexpr int NumThreads = Tx * Ty * Tz;
	constexpr int CopyIterationSize = NumThreads * ElementsPerLoadType;

	constexpr int FiltersSharedElems = ROUND_UP_TO_MULTIPLE_OF(CoutPerBlock * FilterWidth * FilterWidth, CopyIterationSize); // Number of shared memory elements for filters aligned up to copy iteration size
	constexpr int InSharedWidth = Tx + 2 * FilterRadius;
	constexpr int InSharedPitch = ROUND_UP_TO_MULTIPLE_OF(InSharedWidth, ElementsPerLoadType); // Pitch must be divisible by number of elements per LoadType
	constexpr int InSharedHeight = YPerBlock + 2 * FilterRadius;
	constexpr int InSharedElems = InSharedPitch * InSharedHeight;

	constexpr int SharedMemoryElems = ROUND_UP_TO_MULTIPLE_OF(InSharedElems, CopyIterationSize) + FiltersSharedElems; // Since FiltersSharedElems is aligned up to the copy size, and we copy input first, we can safely 

	const int out_width = in_width - 2 * FilterRadius;
	const int out_height = in_height - 2 * FilterRadius;
	const int out_row_pitch = ROUND_UP_TO_MULTIPLE_OF(sizeof(T) * out_width, sizeof(LoadType)) / sizeof(T);
	const u64 out_size = (u64)out_row_pitch * out_height * out_num_channels;

	const int in_row_pitch = ROUND_UP_TO_MULTIPLE_OF(sizeof(T) * in_width, sizeof(LoadType)) / sizeof(T);

	const int filters_block_pitch = ROUND_UP_TO_MULTIPLE_OF(sizeof(T) * FilterWidth * FilterWidth * CoutPerBlock, sizeof(LoadType)) / sizeof(T);
	const int num_blocks_to_fill_out_channels = (out_num_channels + CoutPerBlock - 1) / CoutPerBlock;
	const int filters_in_channel_pitch = filters_block_pitch * num_blocks_to_fill_out_channels;
	const u64 filters_size = ROUND_UP_TO_MULTIPLE_OF(filters_in_channel_pitch * in_num_channels, CopyIterationSize);

	const u32 dim_x = (out_width + Tx - 1) / Tx;
	const u32 dim_y = (out_height + YPerBlock - 1) / YPerBlock;
	const u32 dim_z = (out_num_channels + CoutPerBlock - 1) / CoutPerBlock;

	const dim3 block_size = { Tx, Ty, Tz };
	const dim3 grid_size = { dim_x, dim_y, dim_z };

	constexpr int TotalLoadsNeeded = InSharedElems / ElementsPerLoadType;
	constexpr int LoadTypePerRow = InSharedPitch / ElementsPerLoadType;
	constexpr int ExtraLoads = ((TotalLoadsNeeded + NumThreads - 1) / NumThreads) * NumThreads - TotalLoadsNeeded;
	constexpr int ExtraRows = (ExtraLoads + LoadTypePerRow - 1) / LoadTypePerRow;
	u64 in_size = ROUND_UP_TO_MULTIPLE_OF(
		(u64)in_row_pitch * (u64)in_height * (u64)in_num_channels,
		(YPerBlock * dim_y + 2 * FilterRadius + ExtraRows) * (Tx * dim_x + 2 * FilterRadius)); // Round up the allocation size so that last block will not read out of bounds

	convolution_with_reduction_sizes_t ret;
	ret.grid_size = grid_size;
	ret.block_size = block_size;
	ret.shared_memory_size = SharedMemoryElems;
	ret.input_size = in_size;
	ret.filters_size = filters_size;
	ret.out_size = out_size;
	ret.out_width = out_width;
	ret.out_height = out_height;
	ret.in_row_pitch = in_row_pitch;
	ret.filters_in_channel_pitch = filters_in_channel_pitch;
	ret.filters_block_pitch = filters_block_pitch;
	ret.out_row_pitch = out_row_pitch;
	return ret;
}

template <class T, class LoadType, int FilterRadius, int Tx, int Ty, int Tz, int Rz, int Ry>
convolution_sizes_t get_convolution_sizes(
	int in_num_channels,
	int in_width,
	int in_height,
	int out_num_channels
) {
	constexpr int ElementsPerLoadType = sizeof(LoadType) / sizeof(T);
	constexpr int FilterWidth = FilterRadius * 2 + 1;
	constexpr int CoutPerBlock = Tz * Rz;
	constexpr int YPerBlock = Ry * Ty;
	constexpr int NumThreads = Tx * Ty * Tz;
	constexpr int CopyIterationSize = NumThreads * ElementsPerLoadType;

	constexpr int FiltersSharedElems = ROUND_UP_TO_MULTIPLE_OF(CoutPerBlock * FilterWidth * FilterWidth, CopyIterationSize); // Number of shared memory elements for filters aligned up to copy iteration size
	constexpr int InSharedWidth = Tx + 2 * FilterRadius;
	constexpr int InSharedPitch = ROUND_UP_TO_MULTIPLE_OF(InSharedWidth, ElementsPerLoadType); // Pitch must be divisible by number of elements per LoadType
	constexpr int InSharedHeight = YPerBlock + 2 * FilterRadius;
	constexpr int InSharedElems = InSharedPitch * InSharedHeight;

	constexpr int SharedMemoryElems = ROUND_UP_TO_MULTIPLE_OF(InSharedElems, CopyIterationSize) + FiltersSharedElems; // Since FiltersSharedElems is aligned up to the copy size, and we copy input first, we can safely overwrite with Filters

	const int padded_width = in_width + 2 * FilterRadius;
	const int padded_height = in_height + 2 * FilterRadius;

	const int width = in_width;
	const int height = in_height;

	const int row_pitch = ROUND_UP_TO_MULTIPLE_OF(sizeof(T) * padded_width, sizeof(LoadType)) / sizeof(T);
	const u64 out_size = (u64)row_pitch * padded_height * out_num_channels;

	const int filters_block_pitch = ROUND_UP_TO_MULTIPLE_OF(sizeof(T) * FilterWidth * FilterWidth * CoutPerBlock, sizeof(LoadType)) / sizeof(T);
	const int num_blocks_to_fill_out_channels = (out_num_channels + CoutPerBlock - 1) / CoutPerBlock;
	const int filters_in_channel_pitch = filters_block_pitch * num_blocks_to_fill_out_channels;
	const u64 filters_size =
		ROUND_UP_TO_MULTIPLE_OF(filters_in_channel_pitch * in_num_channels, CopyIterationSize);

	const u32 dim_x = (width + Tx - 1) / Tx;
	const u32 dim_y = (height + YPerBlock - 1) / YPerBlock;
	const u32 dim_z = (out_num_channels + CoutPerBlock - 1) / CoutPerBlock;

	const dim3 block_size = { Tx, Ty, Tz };
	const dim3 grid_size = { dim_x, dim_y, dim_z };

	constexpr int TotalLoadsNeeded = InSharedElems / ElementsPerLoadType;
	constexpr int LoadTypePerRow = InSharedPitch / ElementsPerLoadType;
	constexpr int ExtraLoads = ((TotalLoadsNeeded + NumThreads - 1) / NumThreads) * NumThreads - TotalLoadsNeeded;
	constexpr int ExtraRows = (ExtraLoads + LoadTypePerRow - 1) / LoadTypePerRow;
	const u64 in_size = ROUND_UP_TO_MULTIPLE_OF(
		(u64)row_pitch * (u64)in_height * (u64)in_num_channels,
		(YPerBlock * dim_y + 2 * FilterRadius + ExtraRows) * (Tx * dim_x + 2 * FilterRadius)); // Round up the allocation size so that last block will not read out of bounds

	convolution_sizes_t ret;
	ret.grid_size = grid_size;
	ret.block_size = block_size;
	ret.shared_memory_size = SharedMemoryElems;
	ret.input_size = in_size;
	ret.filters_size = filters_size;
	ret.out_size = out_size;
	ret.filters_in_channel_pitch = filters_in_channel_pitch;
	ret.filters_block_pitch = filters_block_pitch;
	ret.padded_width = padded_width;
	ret.padded_height = padded_height;
	ret.row_pitch = row_pitch;
	return ret;
}

template <class T, int FilterRadius, int Tx, int Ty, int Tz, int Rz>
__global__ void convolution_validation(
	T* __restrict__ input,
	T* __restrict__ filters,
	T* __restrict__ out,
	int in_num_channels,
	int in_height,
	int in_row_pitch,
	int filters_in_channel_pitch,
	int filters_block_pitch,
	int out_num_channels,
	int out_width,
	int out_height,
	int out_row_pitch
) {
	constexpr int FilterWidth = FilterRadius * 2 + 1;

	const int x = threadIdx.x + blockIdx.x * Tx;
	const int y = threadIdx.y + blockIdx.y * Ty;
	const int o = threadIdx.z + blockIdx.z * Tz;

	if (o < out_num_channels && y < out_height && x < out_width) {
		T tmp = 0;
		for (int n = 0; n < in_num_channels; n++) {
			for (int i = 0; i < FilterWidth; i++) {
				for (int j = 0; j < FilterWidth; j++) {
					T inp = input[device_index_arr_of_arr(in_row_pitch, in_height, n, y + i, x + j)];
					T filt = filters[device_index_filters(Rz * Tz, filters_in_channel_pitch, filters_block_pitch, FilterWidth, n, o, i, j)];
					tmp += inp * filt;
				}
			}
		}
		out[device_index_arr_of_arr(out_row_pitch, out_height, o, y, x)] = tmp;
	}
}

template <class T, class LoadType, int FilterRadius, int Tx, int Ty, int Tz, int Rz, int Ry>
void measure_flops(
	FILE* fptr,
	int in_num_channels,
	int in_width,
	int in_height,
	int out_num_channels,
	bool validate
) {
	printf("in_height = %d, in_width = %d\n", in_height, in_width);
	printf("in_num_channels = %d, out_num_channels = %d\n", in_num_channels, out_num_channels);
	constexpr int FilterWidth = (FilterRadius * 2 + 1);

	convolution_with_reduction_sizes_t sizes = get_convolution_with_reduction_sizes<T, LoadType, FilterRadius, Tx, Ty, Tz, Rz, Ry>(in_num_channels, in_width, in_height, out_num_channels);

	// CUDA events for timing
	cudaError_t kernel_error;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	f32 milliseconds = 0;
	f64 flops;

	T* d_filters;
	T* d_input;
	gpuErrchk(cudaMalloc((void**)&d_input, sizes.input_size * sizeof(T)));
	gpuErrchk(cudaMalloc((void**)&d_filters, sizes.filters_size * sizeof(T)));

	T* h_input;
	T* h_filters;
	if (validate) {
		h_filters = (T*)malloc(sizes.filters_size * sizeof(T));
		for (int i = 0; i < sizes.filters_size; i++) {
			*(u32*)(&h_filters[i]) = 0xFFFFFFFF;
		}
		h_input = (T*)malloc(sizes.input_size * sizeof(T));
		for (int i = 0; i < sizes.input_size; i++) {
			*(u32*)&h_input[i] = 0xBBBBBBBB;
		}
		// init input
		for (int n = 0; n < in_num_channels; n++) {
			for (int y = 0; y < in_height; y++) {
				for (int x = 0; x < in_width; x++) {
					h_input[index_arr_of_arr(sizes.in_row_pitch, in_height, n, y, x)] =
						(T)(n * (in_height * in_width) + y * (in_width)+x);
				}
			}
		}

		// init filters
		for (int n = 0; n < in_num_channels; n++) {
			for (int o = 0; o < out_num_channels; o++) {
				for (int y = 0; y < FilterWidth; y++) {
					for (int x = 0; x < FilterWidth; x++) {
						h_filters[index_filters(Rz * Tz, sizes.filters_in_channel_pitch, sizes.filters_block_pitch, FilterWidth, n, o, y, x)] =
							(T)(n * (FilterWidth * FilterWidth * out_num_channels) + o * (FilterWidth * FilterWidth) + y * (FilterWidth)+x);
					}
				}
			}
		}

		// Copy to GPU
		gpuErrchk(cudaMemcpy(d_filters, h_filters, sizes.filters_size * sizeof(T), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_input, h_input, sizes.input_size * sizeof(T), cudaMemcpyHostToDevice));
	}
	else {
		const int init_block_size = 1024;
		u64 in_init_grid_size = (sizes.input_size + (u64)init_block_size - 1) / (u64)init_block_size;
		u64 filters_init_grid_size = (sizes.filters_size + (u64)init_block_size - 1) / (u64)init_block_size;
		device_init_arr<T> << <in_init_grid_size, init_block_size >> > (d_input, 1.0, sizes.input_size);
		device_init_arr<T> << <filters_init_grid_size, init_block_size >> > (d_filters, 1.0, sizes.filters_size);
	}
	cudaDeviceSynchronize();
	cudaError_t init_error = cudaGetLastError();

	T* d_out;
	gpuErrchk(cudaMalloc((void**)&d_out, sizes.out_size * sizeof(T)));

	cudaEventRecord(start);
	constexpr int Iterations = 10;
	for (int i = 0; i < Iterations; i++) {
		convolution_shared_with_reduction<T, LoadType, FilterRadius, Tx, Ty, Tz, Rz, Ry> << < sizes.grid_size, sizes.block_size, sizes.shared_memory_size * sizeof(T) >> > (
			d_input, d_filters, d_out,
			in_num_channels, in_height, sizes.in_row_pitch, sizes.filters_in_channel_pitch, sizes.filters_block_pitch, out_num_channels,
			sizes.out_width, sizes.out_height, sizes.out_row_pitch
			);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	kernel_error = cudaGetLastError();
	cudaEventElapsedTime(&milliseconds, start, stop);
	flops = flops_from_ms(sizes.out_width, sizes.out_height, FilterRadius, in_num_channels, out_num_channels, (milliseconds / (f32)Iterations) / 1000.0);

	const char* format = "%d bit load";
	int needed_size = snprintf(NULL, 0, format, sizeof(LoadType) * 8);
	char* string = (char*)malloc(needed_size + 1);
	snprintf(string, needed_size + 1, format, sizeof(LoadType) * 8); 
	print_FLOPS(string, init_error, kernel_error, fptr, FilterRadius, Tx, Ty, Tz, Rz, Ry, flops);
	free(string);

	fprintf(fptr, "\n");
	printf("\n");

	fflush(fptr);

	if (validate) {
		T* h_out = (T*)malloc(sizes.out_size * sizeof(T));
		T* h_out_validation = (T*)malloc(sizes.out_size * sizeof(T));

		T* d_out_validation; gpuErrchk(cudaMalloc((void**)&d_out_validation, sizeof(T) * sizes.out_size));


		const u32 dim_x = (sizes.out_width + Tx - 1) / Tx;
		const u32 dim_y = (sizes.out_height + Ty - 1) / Ty;
		const u32 dim_z = (out_num_channels + Tz - 1) / Tz;

		const dim3 validation_block_size = { Tx, Ty, Tz };
		const dim3 validation_grid_size = { dim_x, dim_y, dim_z };
		convolution_validation<T, FilterRadius, Tx, Ty, Tz, Rz> << < validation_grid_size, validation_block_size >> > (
			d_input, d_filters, d_out_validation,
			in_num_channels, in_height, sizes.in_row_pitch, sizes.filters_in_channel_pitch, sizes.filters_block_pitch, out_num_channels,
			sizes.out_width, sizes.out_height, sizes.out_row_pitch
		);
		cudaDeviceSynchronize();
		bool hit_error = false;
		cudaMemcpy(h_out, d_out, sizes.out_size * sizeof(T), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_out_validation, d_out_validation, sizes.out_size * sizeof(T), cudaMemcpyDeviceToHost);
		for (int out_ch = 0; out_ch < out_num_channels; out_ch++) {
			for (int y = 0; y < sizes.out_height; y++) {
				for (int x = 0; x < sizes.out_width; x++) {
					T out_value = h_out[index_arr_of_arr(sizes.out_row_pitch, sizes.out_height, out_ch, y, x)];
					T validation_out_value = h_out_validation[index_arr_of_arr(sizes.out_row_pitch, sizes.out_height, out_ch, y, x)];
					if (!(out_value + 0.0000001f >= validation_out_value && out_value - 0.0000001f <= validation_out_value)) {
						printf("error, out[%d, %d, %d] had value %f, validation had value %f, diff = %f\n", out_ch, y, x, out_value, validation_out_value, validation_out_value - out_value);
						hit_error = true;
					}
				}
			}
		}
		if (!hit_error) {
			printf("Valid, bro.\n----------------\n\n");
		}

		free(h_out);
		free(h_out_validation);
		free(h_input);
		free(h_filters);
		cudaFree(d_out_validation);
	}

	// Clean up CUDA events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_input);
	cudaFree(d_filters);
	cudaFree(d_out);
}

template <class T, class LoadType, int FilterRadius, int Tx, int Ty, int Tz, int Rz, int Ry>
void measure_flops_no_reduction(
	FILE* fptr,
	int in_num_channels,
	int in_width,
	int in_height,
	int out_num_channels
) {
	convolution_sizes_t sizes = get_convolution_sizes<T, LoadType, FilterRadius, Tx, Ty, Tz, Rz, Ry>(in_num_channels, in_width, in_height, out_num_channels);

	// CUDA events for timing
	cudaError_t kernel_error;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	f32 milliseconds = 0;
	f64 flops;

	T* d_filters;
	T* d_input;
	gpuErrchk(cudaMalloc((void**)&d_input, sizes.input_size * sizeof(T)));
	gpuErrchk(cudaMalloc((void**)&d_filters, sizes.filters_size * sizeof(T)));
	const int init_block_size = 1024;
	u64 in_init_grid_size = (sizes.input_size + (u64)init_block_size - 1) / (u64)init_block_size;
	u64 filters_init_grid_size = (sizes.filters_size + (u64)init_block_size - 1) / (u64)init_block_size;
	device_init_arr<T> << <in_init_grid_size, init_block_size >> > (d_input, 1.0, sizes.input_size);
	device_init_arr<T> << <filters_init_grid_size, init_block_size >> > (d_filters, 1.0, sizes.filters_size);
	cudaDeviceSynchronize();
	cudaError_t init_error = cudaGetLastError();

	T* h_input = (T*)malloc(sizes.input_size * sizeof(T));
	T* h_filters = (T*)malloc(sizes.input_size * sizeof(T));
	cudaMemcpy(h_filters, d_filters, sizes.filters_size * sizeof(T), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_input, d_input, sizes.filters_size * sizeof(T), cudaMemcpyDeviceToHost);

	T* d_out;
	gpuErrchk(cudaMalloc((void**)&d_out, sizes.out_size * sizeof(T)));

	cudaEventRecord(start);
	constexpr int Iterations = 10;
	for (int i = 0; i < Iterations; i++) {
		convolution_shared<T, LoadType, FilterRadius, Tx, Ty, Tz, Rz, Ry> << < sizes.grid_size, sizes.block_size, sizes.shared_memory_size * sizeof(T) >> > (
			d_input, d_filters, d_out, in_num_channels, sizes.padded_height, sizes.padded_width, sizes.row_pitch,
			sizes.filters_in_channel_pitch, sizes.filters_block_pitch, out_num_channels
			);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	kernel_error = cudaGetLastError();
	cudaEventElapsedTime(&milliseconds, start, stop);
	flops = flops_from_ms(in_width, in_height, FilterRadius, in_num_channels, out_num_channels, (milliseconds / (f32)Iterations) / 1000.0);


	const char* format = "%d bit load";
	int needed_size = snprintf(NULL, 0, format, sizeof(LoadType) * 8); 
	char* string = (char*)malloc(needed_size + 1);
	snprintf(string, needed_size + 1, format, sizeof(LoadType) * 8);
	print_FLOPS(string, init_error, kernel_error, fptr, FilterRadius, Tx, Ty, Tz, Rz, Ry, flops);
	free(string);

	fprintf(fptr, "\n");
	printf("\n");

	fflush(fptr);

	// T* h_out = (T*)malloc(sizes.out_size * sizeof(T));
	// cudaMemcpy(h_out, d_out, sizes.out_size * sizeof(T), cudaMemcpyDeviceToHost);

	// Clean up CUDA events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_input);
	cudaFree(d_filters);
	cudaFree(d_out);
}

template<class T>
void run_timed() {
	FILE* fptr = fopen("result.txt", "w");
	fptr = freopen("result.txt", "a", fptr);

	printf("----warmup----\n");
	for (int i = 0; i < 5; i++) {
		measure_flops<T, float4, 1, 32, 2, 2, 4, 13>(fptr, 32, 4096, 4096, 32, false);
	}
	printf("----end of warmup----\n");
	
	measure_flops<T, float4, 1, 32, 2, 2, 4, 13>(fptr, 32, 4096, 4096, 32, false); // 1.761826e+13 FLOPS
	measure_flops<T, float4, 2, 32, 2, 2, 4, 10>(fptr, 32, 4096, 4096, 32, false); // 1.833579e+13 FLOPS
	measure_flops<T, float4, 3, 32, 2, 2, 2, 16>(fptr, 32, 4096, 4096, 32, false); // 1.853640e+13 FLOPS
	measure_flops<T, float4, 4, 32, 2, 2, 4,  8>(fptr, 32, 4096, 4096, 32, false); // 1.872913e+13 FLOPS
	measure_flops<T, float4, 5, 32, 2, 2, 4,  6>(fptr, 32, 4096, 4096, 32, false); // 1.817919e+13 FLOPS
	measure_flops<T, float4, 6, 32, 2, 2, 2,  5>(fptr, 32, 4096, 4096, 32, false); // 1.799700e+13 FLOPS
	measure_flops<T, float4, 7, 32, 2, 2, 2,  5>(fptr, 32, 4096, 4096, 32, false); // 1.766944e+13 FLOPS
	measure_flops<T, float4, 8, 32, 2, 2, 2,  8>(fptr, 32, 4096, 4096, 32, false); // 1.737954e+13 FLOPS

	// Validation
#if 1
	measure_flops<T, float4, 1, 32, 2, 2, 4, 13>(fptr, 32/4, 512, 512, 32/4, true); // 1.761826e+13 FLOPS
	measure_flops<T, float4, 1, 32, 2, 2, 4, 13>(fptr, 1, 500, 500, 11, true); // 1.761826e+13 FLOPS
	measure_flops<T, float4, 1, 32, 2, 1, 1, 13>(fptr, 1, 100, 300, 1, true); // 1.761826e+13 FLOPS
	measure_flops<T, float4, 1, 32, 2, 2, 4, 13>(fptr, 32, 2048, 2048, 32, true); // 1.761826e+13 FLOPS
	measure_flops<T, float4, 1, 32, 2, 2, 4, 13>(fptr, 22, 500, 500, 22, true); // 1.761826e+13 FLOPS
	measure_flops<T, float4, 1, 32, 2, 1, 1, 13>(fptr, 1, 2048, 2048, 32, true); // 1.761826e+13 FLOPS
#endif

	fclose(fptr);
}

int main(int argc, char* argv[]) {

	run_timed<float>();
}