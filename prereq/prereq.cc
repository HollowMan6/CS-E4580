#include <cmath>
#include <cstdint>
struct Result
{
    float avg[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- horizontal position: 0 <= x0 < x1 <= nx
- vertical position: 0 <= y0 < y1 <= ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
- output: avg[c]
*/
Result calculate(int ny, int nx, const float *data, int y0, int x0, int y1, int x1)
{
    Result result{{0.0f, 0.0f, 0.0f}};

    int num_pixels = (y1 - y0) * (x1 - x0);
    double r_sum = 0.0;
    double g_sum = 0.0;
    double b_sum = 0.0;

    for (int y = y0; y < y1; ++y)
    {
        for (int x = x0; x < x1; ++x)
        {
            r_sum += data[3 * (y * nx + x) + 0];
            g_sum += data[3 * (y * nx + x) + 1];
            b_sum += data[3 * (y * nx + x) + 2];
        }
    }

    result.avg[0] = r_sum / num_pixels;
    result.avg[1] = g_sum / num_pixels;
    result.avg[2] = b_sum / num_pixels;

    return result;
}
