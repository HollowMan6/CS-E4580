#include <cmath>
#include <vector>
typedef double double4_t __attribute__((vector_size(4 * sizeof(double))));

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
    const int nx_v = (nx + element - 1) / element;
    const int ny_v = (ny + element - 1) / element;
    std::vector<std::vector<double4_t>> data_copy_v(ny_v * element, std::vector<double4_t>(nx_v));

#pragma omp parallel for
    for (int y = 0; y < ny; y++)
    {
        for (int x = 0; x < nx_v; x++)
        {
            const int base = x * element;
            double4_t v = {0.0, 0.0, 0.0, 0.0};
            for (int e = 0; e < element && base + e < nx; e++)
            {
                v[e] = data_copy[y *nx + base + e];
            }
            data_copy_v[y][x] = v;
        }
    }

#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < ny_v * element; i += element)
    {
        for (int j = i; j < ny_v * element; j += element)
        {
            double4_t sums[element][element] = {{0.0}};

            for (int k = 0; k < nx_v; k++)
            {
                const double4_t x0 = data_copy_v[i][k];
                const double4_t x1 = data_copy_v[i + 1][k];
                const double4_t x2 = data_copy_v[i + 2][k];
                const double4_t x3 = data_copy_v[i + 3][k];

                const double4_t y0 = data_copy_v[j][k];
                const double4_t y1 = data_copy_v[j + 1][k];
                const double4_t y2 = data_copy_v[j + 2][k];
                const double4_t y3 = data_copy_v[j + 3][k];
                sums[0][0] += x0 * y0;
                sums[0][1] += x0 * y1;
                sums[0][2] += x0 * y2;
                sums[0][3] += x0 * y3;

                sums[1][0] += x1 * y0;
                sums[1][1] += x1 * y1;
                sums[1][2] += x1 * y2;
                sums[1][3] += x1 * y3;

                sums[2][0] += x2 * y0;
                sums[2][1] += x2 * y1;
                sums[2][2] += x2 * y2;
                sums[2][3] += x2 * y3;

                sums[3][0] += x3 * y0;
                sums[3][1] += x3 * y1;
                sums[3][2] += x3 * y2;
                sums[3][3] += x3 * y3;
            }

            for (int av = 0; av < element; av++)
            {
                for (int bv = 0; bv < element; bv++)
                {
                    int a = i + av;
                    int b = j + bv;
                    const double4_t c = sums[av][bv];
                    if (i < ny - av && j < ny - bv)
                        result[ny * a + b] = c[0] + c[1] + c[2] + c[3];
                }
            }
        }
    }

    delete[] data_copy;
}
