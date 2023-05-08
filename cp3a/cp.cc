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
    std::vector<std::vector<double> > data_copy(ny, std::vector<double>(nx));

    #pragma omp parallel for
    for (int y = 0; y < ny; y++)
    {
        // step 1: normalize rows to have mean 0
        double mean = 0;
        for (int x = 0; x < nx; x++)
        {
            mean += data[x + y * nx];
        }
        mean /= nx;
        double scale = 0;
        for (int x = 0; x < nx; x++)
        {
            data_copy[y][x] = data[x + y * nx] - mean;
            scale += data_copy[y][x] * data_copy[y][x];
        }

        // step 2: normalize rows to have length 1
        scale = sqrt(scale);
        if (scale == 0)
        {
            scale = 1;
        }
        for (int x = 0; x < nx; x++)
        {
            data_copy[y][x] /= scale;
        }
    }

    // step 3: calculate matrix product
    int element = 4;
    int nx_v = (nx + element - 1) / element;
    std::vector<std::vector<double4_t> > data_copy_v(ny, std::vector<double4_t>(nx_v));

    #pragma omp parallel for
    for (int y = 0; y < ny; y++)
    {
        for (int x = 0; x < nx_v; x++)
        {
            for (int e = 0; e < element; e++)
            {
                int i = x * element + e;
                data_copy_v[y][x][e] = i < nx ? data_copy[y][i] : 0.0;
            }
        }
    }

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < ny; y++)
    {
        result[y * ny + y] = 1.0;
        for (int x = y + 1; x < ny; x++)
        {
            double4_t sum = {0.0, 0.0, 0.0, 0.0};
            for (int v = 0; v < nx_v; v++)
                sum += data_copy_v[y][v] * data_copy_v[x][v];

            double dot = 0.0;
            for (int i = 0; i < 4; i++)
                dot += sum[i];
            result[x + y * ny] = dot;
        }
    }
}
