#include <cstdio>

#define BLOCK_SIZE	16
#define N		1024
#define ELEMENT_NUM	1048576             /* 1048576 = 1024 * 1024, the capacity of the matrix */

#include "tools/random.h"
#include "tools/test.h"
#include "tools/timing.h"

__global__ void trans(int *addr_src, int *addr_des) {
	__shared__ int cache[BLOCK_SIZE][BLOCK_SIZE];   /* Allocate shared memory as cache */
	int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;

	cache[threadIdx.y][threadIdx.x] = addr_src[row * N + col];  /* Copy data from global memory to cache */

	__syncthreads();

	/* Renew row and col */
	col = blockIdx.y * BLOCK_SIZE + threadIdx.x;
	row = blockIdx.x * BLOCK_SIZE + threadIdx.y;

	addr_des[row * N + col] = cache[threadIdx.x][threadIdx.y];  /* Copy data from cache to global memory */
}

int main() {
	int size = sizeof (int) * ELEMENT_NUM;  /* Number of bytes for 1024 * 1024 integers */

	/* Allocate memory in DRAM */
	int *host_src = (int *)malloc(size);
	int *host_des = (int *)malloc(size);

	set_random(host_src);               /* Set random value for each element in source matrix */

	/* Allocate global memory on GPU */
	int *dev_src, *dev_des;
	cudaMalloc(&dev_src, size);
	cudaMalloc(&dev_des, size);

	cudaMemcpy(dev_src, host_src, size, cudaMemcpyHostToDevice);    /* Copy data from DRAM to GPU */

	Timing timer;                                                   /* A timer which can help record the running time */

	/* Configure parameters of grid and block */
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim(N / blockDim.x, N / blockDim.y);

	/* Execute transposition on GPU and record the running time */
	timer.beginTiming();
	trans << < gridDim, blockDim >> > (dev_src, dev_des);
	timer.endTiming();

	cudaMemcpy(host_des, dev_des, size, cudaMemcpyDeviceToHost);    /* Copy data from GPU to DRAM */

	timer.printTime();                                              /* Print out running time */
	Test(host_src, host_des);                                       /* Test whether the result is correct */

	/* Free the allocated memories */
	cudaFree(dev_src);
	cudaFree(dev_des);
	free(host_src);
	free(host_des);

	return 0;
}
