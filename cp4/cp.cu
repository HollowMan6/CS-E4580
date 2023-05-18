#include <cmath>
#include <vector>

#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

static inline void check(cudaError_t err, const char *context)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << context << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static inline int divup(int a, int b)
{
    return (a + b - 1) / b;
}

#define CHECK(x) check(x, #x)

__global__ void kernel(const int nx, const int ny, float *norm, float *result)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= ny || j >= ny)
        return;

    float sum = 0.0;
    for (int k = 0; k < nx; k++)
    {
        sum += norm[i * nx + k] * norm[j * nx + k];
    }
    result[j * ny + i] = sum;
}

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result)
{
    std::vector<float> data_copy(ny * nx);

    for (int y = 0; y < ny; y++)
    {
        // step 1: normalize rows to have mean 0
        float mean = 0;
        for (int x = 0; x < nx; x++)
        {
            mean += data[x + y * nx];
        }
        mean /= nx;
        float scale = 0;
        for (int x = 0; x < nx; x++)
        {
            data_copy[y * nx + x] = data[y * nx + x] - mean;
            scale += data_copy[y * nx + x] * data_copy[y * nx + x];
        }

        // step 2: normalize rows to have length 1
        scale = sqrt(scale);
        if (scale == 0)
        {
            scale = 1;
        }
        for (int x = 0; x < nx; x++)
        {
            data_copy[y * nx + x] /= scale;
        }
    }

    float *GPU_n = NULL;
    CHECK(cudaMalloc((void **)&GPU_n, ny * nx * sizeof(float)));
    CHECK(cudaMemcpy(GPU_n, data_copy.data(), ny * nx * sizeof(float), cudaMemcpyHostToDevice));

    float *GPU_r = NULL;
    CHECK(cudaMalloc((void **)&GPU_r, ny * ny * sizeof(float)));
    CHECK(cudaMemset(GPU_r, 0, ny * ny * sizeof(float)));

    dim3 dim_block(8, 8);
    dim3 dim_grid(divup(ny, dim_block.x), divup(ny, dim_block.y));
    kernel<<<dim_grid, dim_block>>>(nx, ny, GPU_n, GPU_r);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(result, GPU_r, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(GPU_r));
    CHECK(cudaFree(GPU_n));
}
