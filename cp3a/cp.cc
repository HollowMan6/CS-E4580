#include <cmath>
#include <vector>
#include <x86intrin.h>

typedef double double4_t __attribute__((vector_size(4 * sizeof(double))));
static inline double4_t swap2(double4_t x) { return _mm256_permute2f128_pd(x, x, 0b00000001); }
static inline double4_t swap1(double4_t x) { return _mm256_permute_pd(x, 0b00000101); }

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
    double *data_copy = new double[ny * nx];

#pragma omp parallel for
    for (int y = 0; y < ny; y++)
    {
        // step 1: normalize rows to have mean 0
        double sum = 0;
        for (int x = 0; x < nx; x++)
        {
            sum += data[x + y * nx];
        }
        double mean = sum / nx;
        double scale = 0;
        for (int x = 0; x < nx; x++)
        {
            data_copy[x + y * nx] = data[x + y * nx] - mean;
            scale += data_copy[x + y * nx] * data_copy[x + y * nx];
        }

        // step 2: normalize rows to have length 1
        scale = sqrt(scale);
        if (scale == 0)
        {
            scale = 1;
        }
        for (int x = 0; x < nx; x++)
        {
            data_copy[x + y * nx] /= scale;
        }
    }

    // step 3: calculate matrix product
    const int element = 4;
    const int ny_v = (ny + element - 1) / element;
    std::vector<std::vector<double4_t>> data_copy_v(ny_v, std::vector<double4_t>(nx));

#pragma omp parallel for
    for (int y = 0; y < ny_v; y++)
    {
        for (int x = 0; x < nx; x++)
        {
            const int base = y * element;
            double4_t v = {0.0, 0.0, 0.0, 0.0};
            for (int e = 0; e < element && base + e < ny; e++)
            {
                v[e] = data_copy[nx * (base + e) + x];
            }
            data_copy_v[y][x] = v;
        }
    }

#pragma omp parallel for schedule(static, 1)
    for (int x = 0; x < ny_v; ++x)
    {
        for (int y = x; y < ny_v; ++y)
        {
            double4_t sums00 = {0.0, 0.0, 0.0, 0.0};
            double4_t sums01 = {0.0, 0.0, 0.0, 0.0};
            double4_t sums10 = {0.0, 0.0, 0.0, 0.0};
            double4_t sums11 = {0.0, 0.0, 0.0, 0.0};

            for (int k = 0; k < nx; ++k)
            {
                double4_t x00 = data_copy_v[x][k];
                double4_t y00 = data_copy_v[y][k];
                double4_t x10 = swap2(x00);
                double4_t y01 = swap1(y00);

                sums00 += x00 * y00;
                sums01 += x00 * y01;
                sums10 += x10 * y00;
                sums11 += x10 * y01;
            }

            double4_t sums[4] = {sums00, sums01, sums10, sums11};
            for (int e = 1; e < element; e += 2)
            {
                sums[e] = swap1(sums[e]);
            }

            for (int a = 0; a < element; a++)
            {
                for (int b = 0; b < element; b++)
                {
                    int i = a + x * element;
                    int j = b + y * element;
                    if (i <= j)
                    {
                        if (j < ny && i < ny)
                        {
                            result[ny * i + j] = sums[a ^ b][b];
                        }
                    }
                }
            }
        }
    }

    delete[] data_copy;
}
