#include <cmath>
#include <vector>
#include <immintrin.h>

typedef double double8_t __attribute__((vector_size(8 * sizeof(double))));

static inline double8_t swap4(double8_t x) { return _mm512_permutexvar_pd(_mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4), x); }
static inline double8_t swap2(double8_t x) { return _mm512_permutex_pd(x, 0b01001110); }
static inline double8_t swap1(double8_t x) { return _mm512_permute_pd(x, 0b01010101); }

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
    // step 3: calculate matrix product
    const int element = 8;
    const int ny_v = (ny + element - 1) / element;

    const int nx_v = (nx + element - 1) / element;
    const int nx_vc = nx_v * element;
    const int ny_vbr = (ny_v + element - 1) / element;
    const int ny_vr = ny_vbr * element;

    const double8_t zero = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    std::vector<std::vector<double8_t>> data_copy_v(ny_vr, std::vector<double8_t>(nx_vc));

#pragma omp parallel for
    for (int y = 0; y < ny_v; y++)
    {
        for (int x = 0; x < nx; x++)
        {
            const int base = y * element;
            double8_t v = zero;
            for (int e = 0; e < element && base + e < ny; e++)
            {
                v[e] = data[nx * (base + e) + x];
            }
            data_copy_v[y][x] = v;
        }
    }

    double div_nx = 1 / (double)nx;
    double8_t div_nx_v = {div_nx, div_nx, div_nx, div_nx, div_nx, div_nx, div_nx, div_nx};

#pragma omp parallel for
    for (int y = 0; y < ny_v; y++)
    {
        double8_t sum = zero;
        double8_t sqr = zero;
        double8_t norm = zero;

        // step 1: normalize rows to have mean 0
        for (int x = 0; x < nx; x++)
        {
            sum += data_copy_v[y][x];
        }
        double8_t mean = sum * div_nx_v;
        for (int x = 0; x < nx; x++)
        {
            double8_t num = data_copy_v[y][x];
            double8_t norm = num - mean;
            data_copy_v[y][x] = norm;
            sqr += norm * norm;
        }
        // step 2: normalize rows to have length 1
        sqr *= div_nx_v;
        for (int e = 0; e < element; e++)
        {
            if (sqr[e] != 0)
                norm[e] = 1 / std::sqrt(sqr[e]);
        }
        for (int x = 0; x < nx; x++)
        {
            data_copy_v[y][x] *= norm;
        }
    }

// step 3: calculate matrix product
#pragma omp parallel for schedule(dynamic, 1)
    for (int x = 0; x < ny_vbr; x++)
    {
        for (int y = x; y < ny_vbr; y++)
        {
            double8_t sums[element][element][element] = {{{0.0}}};
            for (int m = 0; m < nx_v; m++)
            {
                for (int a = 0; a < element; a++)
                {
                    for (int b = 0; b < element; b++)
                    {
                        int j = x * element + a;
                        int i = y * element + b;
                        double8_t sum000 = zero;
                        double8_t sum001 = zero;
                        double8_t sum010 = zero;
                        double8_t sum011 = zero;
                        double8_t sum100 = zero;
                        double8_t sum101 = zero;
                        double8_t sum110 = zero;
                        double8_t sum111 = zero;
                        for (int k = 0; k < element; k++)
                        {
                            double8_t x000 = data_copy_v[j][k + m * element];
                            double8_t y000 = data_copy_v[i][k + m * element];
                            double8_t x100 = swap4(x000);
                            double8_t x010 = swap2(x000);
                            double8_t x110 = swap2(x100);
                            double8_t y001 = swap1(y000);
                            sum000 += x000 * y000;
                            sum001 += x000 * y001;
                            sum010 += x010 * y000;
                            sum011 += x010 * y001;
                            sum100 += x100 * y000;
                            sum101 += x100 * y001;
                            sum110 += x110 * y000;
                            sum111 += x110 * y001;
                        }
                        sums[a][b][0] += sum000 * div_nx_v;
                        sums[a][b][1] += sum001 * div_nx_v;
                        sums[a][b][2] += sum010 * div_nx_v;
                        sums[a][b][3] += sum011 * div_nx_v;
                        sums[a][b][4] += sum100 * div_nx_v;
                        sums[a][b][5] += sum101 * div_nx_v;
                        sums[a][b][6] += sum110 * div_nx_v;
                        sums[a][b][7] += sum111 * div_nx_v;
                    }
                }
            }
            for (int a = 0; a < element; a++)
            {
                for (int b = 0; b < element; b++)
                {
                    for (int kb = 1; kb < element; kb += 2)
                    {
                        sums[a][b][kb] = swap1(sums[a][b][kb]);
                    }
                    for (int jb = 0; jb < element; jb++)
                    {
                        for (int ib = 0; ib < element; ib++)
                        {
                            int j = x * element * element + a * element + ib;
                            int i = y * element * element + b * element + jb;
                            if (j < ny && i < ny)
                            {
                                result[j * ny + i] = sums[a][b][ib ^ jb][jb];
                            }
                        }
                    }
                }
            }
        }
    }
}
