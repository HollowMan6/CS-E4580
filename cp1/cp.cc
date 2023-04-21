#include <cmath>

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
    for (int i = 0; i < ny; i++)
    {
        for (int j = i; j < ny; j++)
        {
            double dot = 0;
            for (int x = 0; x < nx; x++)
            {
                dot += data_copy[x + i * nx] * data_copy[x + j * nx];
            }
            result[j + i * ny] = dot;
        }
    }
    delete[] data_copy;
}
