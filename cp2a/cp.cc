#include <cmath>
#include <vector>
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
    std::vector<std::vector<double>> data_copy(ny, std::vector<double>(nx));
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
    for (int i = 0; i < ny; i++)
    {
        for (int j = i; j < ny; j++)
        {
            double dot[4] = {0.0};
            for (int x = 0; x <= nx / 4; x++)
            {
                for (int k = 0; k < 4 && x * 4 + k < nx; k++)
                {
                    dot[k] += data_copy[i][4 * x + k] * data_copy[j][4 * x + k];
                }
            }
            result[j + i * ny] = dot[0] + dot[1] + dot[2] + dot[3];
        }
    }
}
