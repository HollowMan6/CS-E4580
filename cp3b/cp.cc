#include <cmath>
#include <vector>
typedef float float8_t __attribute__((vector_size(8 * sizeof(float))));

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
    float *data_copy = new float[ny * nx];

#pragma omp parallel for
    for (int y = 0; y < ny; y++)
    {
        // step 1: normalize rows to have mean 0
        float sum = 0;
        for (int x = 0; x < nx; x++)
        {
            sum += data[x + y * nx];
        }
        float mean = sum / nx;
        float scale = 0;
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
    const int element = 8;
    const int nx_v = (nx + element - 1) / element;
    const int ny_v = (ny + element - 1) / element;
    std::vector<std::vector<float8_t>> data_copy_v(ny_v * element, std::vector<float8_t>(nx_v));

#pragma omp parallel for
    for (int y = 0; y < ny; y++)
    {
        for (int x = 0; x < nx_v; x++)
        {
            const int base = x * element;
            float8_t v = {0.0, 0.0, 0.0, 0.0};
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
            std::vector<std::vector<float8_t>> sums(element, std::vector<float8_t>(element));

            for (int k = 0; k < nx_v; k++)
            {
                const float8_t x0 = data_copy_v[i][k];
                const float8_t x1 = data_copy_v[i + 1][k];
                const float8_t x2 = data_copy_v[i + 2][k];
                const float8_t x3 = data_copy_v[i + 3][k];
                const float8_t x4 = data_copy_v[i + 4][k];
                const float8_t x5 = data_copy_v[i + 5][k];
                const float8_t x6 = data_copy_v[i + 6][k];
                const float8_t x7 = data_copy_v[i + 7][k];

                const float8_t y0 = data_copy_v[j][k];
                const float8_t y1 = data_copy_v[j + 1][k];
                const float8_t y2 = data_copy_v[j + 2][k];
                const float8_t y3 = data_copy_v[j + 3][k];
                const float8_t y4 = data_copy_v[j + 4][k];
                const float8_t y5 = data_copy_v[j + 5][k];
                const float8_t y6 = data_copy_v[j + 6][k];
                const float8_t y7 = data_copy_v[j + 7][k];

                sums[0][0] += x0 * y0;
                sums[0][1] += x0 * y1;
                sums[0][2] += x0 * y2;
                sums[0][3] += x0 * y3;
                sums[0][4] += x0 * y4;
                sums[0][5] += x0 * y5;
                sums[0][6] += x0 * y6;
                sums[0][7] += x0 * y7;

                sums[1][0] += x1 * y0;
                sums[1][1] += x1 * y1;
                sums[1][2] += x1 * y2;
                sums[1][3] += x1 * y3;
                sums[1][4] += x1 * y4;
                sums[1][5] += x1 * y5;
                sums[1][6] += x1 * y6;
                sums[1][7] += x1 * y7;

                sums[2][0] += x2 * y0;
                sums[2][1] += x2 * y1;
                sums[2][2] += x2 * y2;
                sums[2][3] += x2 * y3;
                sums[2][4] += x2 * y4;
                sums[2][5] += x2 * y5;
                sums[2][6] += x2 * y6;
                sums[2][7] += x2 * y7;

                sums[3][0] += x3 * y0;
                sums[3][1] += x3 * y1;
                sums[3][2] += x3 * y2;
                sums[3][3] += x3 * y3;
                sums[3][4] += x3 * y4;
                sums[3][5] += x3 * y5;
                sums[3][6] += x3 * y6;
                sums[3][7] += x3 * y7;

                sums[4][0] += x4 * y0;
                sums[4][1] += x4 * y1;
                sums[4][2] += x4 * y2;
                sums[4][3] += x4 * y3;
                sums[4][4] += x4 * y4;
                sums[4][5] += x4 * y5;
                sums[4][6] += x4 * y6;
                sums[4][7] += x4 * y7;

                sums[5][0] += x5 * y0;
                sums[5][1] += x5 * y1;
                sums[5][2] += x5 * y2;
                sums[5][3] += x5 * y3;
                sums[5][4] += x5 * y4;
                sums[5][5] += x5 * y5;
                sums[5][6] += x5 * y6;
                sums[5][7] += x5 * y7;

                sums[6][0] += x6 * y0;
                sums[6][1] += x6 * y1;
                sums[6][2] += x6 * y2;
                sums[6][3] += x6 * y3;
                sums[6][4] += x6 * y4;
                sums[6][5] += x6 * y5;
                sums[6][6] += x6 * y6;
                sums[6][7] += x6 * y7;

                sums[7][0] += x7 * y0;
                sums[7][1] += x7 * y1;
                sums[7][2] += x7 * y2;
                sums[7][3] += x7 * y3;
                sums[7][4] += x7 * y4;
                sums[7][5] += x7 * y5;
                sums[7][6] += x7 * y6;
                sums[7][7] += x7 * y7;
            }

            for (int av = 0; av < element; av++)
            {
                for (int bv = 0; bv < element; bv++)
                {
                    int a = i + av;
                    int b = j + bv;
                    const float8_t c = sums[av][bv];
                    if (i < ny - av && j < ny - bv)
                        result[ny * a + b] = c[0] + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7];
                }
            }
        }
    }

    delete[] data_copy;
}
