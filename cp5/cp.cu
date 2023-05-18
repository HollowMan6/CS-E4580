#include <vector>
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

static inline int roundup(int a, int b)
{
    return divup(a, b) * b;
}

#define CHECK(x) check(x, #x)

__device__ float warp(float val)
{

    for (int i = 1; i < 32; i *= 2)
        val += __shfl_xor_sync(0xffffffff, val, i);
    return val;
}

__global__ void kernel_p(float *norm, const float *data, const int nx, const int nx_p, const int ny, const int ny_p)
{
    int it = threadIdx.x;
    int j = blockIdx.y;

    if (ny <= j)
        return;

    float row_sum = 0, sq_sum = 0;
    float *t = norm + nx_p * ny_p;

    for (int ia = 0; ia < nx_p; ia += 32)
    {
        int i = it + ia;
        float v = (i < nx) ? data[j * nx + i] : 0;
        row_sum += v;
    }
    __syncthreads();

    row_sum = warp(row_sum);

    for (int ia = 0; ia < nx_p; ia += 32)
    {
        int i = it + ia;
        norm[j * nx_p + i] = (i < nx) ? (data[j * nx + i] - row_sum / nx) : 0;
        sq_sum += norm[j * nx_p + i] * norm[j * nx_p + i];
    }
    __syncthreads();

    sq_sum = warp(sq_sum);

    for (int ia = 0; ia < nx_p; ia += 32)
    {
        int i = it + ia;
        float v = i < nx ? norm[j * nx_p + i] / sqrt(sq_sum) : 0;
        norm[j * nx_p + i] = v;
        t[i * ny_p + j] = v;
    }
}

__global__ void kernel_dot(const float *norm, float *result, const int nx, const int nx_p, const int ny, const int ny_p)
{
    int bi = blockIdx.x;
    int bj = blockIdx.y;
    int ti = threadIdx.x;
    int tj = threadIdx.y;

    if (bi < bj || ti + bi * blockDim.x > ny || tj + bj * blockDim.y > ny)
        return;

    const float *norm_t = norm + nx_p * ny_p;
    float v[8][8];
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            v[i][j] = 0;
        }
    }
    float x[8];
    float y[8];
    for (int k = 0; k < nx; ++k)
    {

        for (int b = 0; b < 8; ++b)
        {
            int i = bi * 64 + b * 8 + ti;
            int j = bj * 64 + b * 8 + tj;
            x[b] = norm_t[ny_p * k + i];
            y[b] = norm_t[ny_p * k + j];
        }
        for (int vj = 0; vj < 8; ++vj)
        {
            for (int vi = 0; vi < 8; ++vi)
            {
                v[vj][vi] += x[vi] * y[vj];
            }
        }
    }

    for (int lj = 0; lj < 8; ++lj)
    {
        int j = bj * 64 + lj * 8 + tj;
        for (int li = 0; li < 8; ++li)
        {
            int i = bi * 64 + li * 8 + ti;

            if (i < ny && j < ny)
            {
                result[ny * j + i] = v[lj][li];
            }
        }
    }
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
    int nx_p = roundup(nx, 64);
    int ny_p = roundup(ny, 64);

    float *gpu = NULL;
    CHECK(cudaMalloc((void **)&gpu, ny * nx * sizeof(float)));
    CHECK(cudaMemcpy(gpu, data, ny * nx * sizeof(float), cudaMemcpyHostToDevice));

    float *gpu_n = NULL;
    CHECK(cudaMalloc((void **)&gpu_n, 2 * ny_p * nx_p * sizeof(float)));

    float *gpu_t = gpu_n + ny_p * nx_p;
    CHECK(cudaMemset(gpu_t, 0, ny_p * nx_p * sizeof(float)));

    float *gpu_r = NULL;
    CHECK(cudaMalloc((void **)&gpu_r, ny * ny * sizeof(float)));
    CHECK(cudaMemset(gpu_r, 0, ny * ny * sizeof(float)));

    dim3 dim_b(32, 1);
    dim3 dim_g(1, ny_p);
    kernel_p<<<dim_g, dim_b>>>(gpu_n, gpu, nx, nx_p, ny, ny_p);
    CHECK(cudaGetLastError());

    dim_b = dim3(8, 8);
    dim_g = dim3(ny_p / 64, ny_p / 64);
    kernel_dot<<<dim_g, dim_b>>>(gpu_n, gpu_r, nx, nx_p, ny, ny_p);

    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(result, gpu_r, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(gpu_r));
    CHECK(cudaFree(gpu_n));
    CHECK(cudaFree(gpu));
}
