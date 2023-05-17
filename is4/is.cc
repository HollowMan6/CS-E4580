#include <vector>

typedef double double4_t __attribute__((vector_size(4 * sizeof(double))));

double4_t zero{0.0, 0.0, 0.0, 0.0};

struct Result
{
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data)
{
    std::vector<std::vector<double4_t>> data_v(ny + 1, std::vector<double4_t>(nx + 1));

#pragma omp parallel for schedule(static, 1)
    for (int y = 0; y <= ny; y++)
    {
        for (int x = 0; x <= nx; x++)
        {
            data_v[y][x] = zero;
        }
    }

#pragma omp parallel for schedule(static, 1)
    for (int y = 1; y <= ny; y++)
    {
        for (int x = 1; x <= nx; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                data_v[y][x][c] = data[c + 3 * (x - 1) + 3 * nx * (y - 1)] + data_v[y][x - 1][c];
            }
        }
    }

#pragma omp parallel for schedule(static, 1)
    for (int x = 1; x <= nx; x++)
    {
        for (int y = 1; y <= ny; y++)
        {
            data_v[y][x] += data_v[y - 1][x];
        }
    }

    double4_t sum_p = data_v[ny][nx];
    double size_p = ny * nx;

    int y0_r = 0;
    int x0_r = 0;
    int y1_r = 0;
    int x1_r = 0;
    double u = -1;
#pragma omp parallel
    {
        double u_t = -1;
        int x0_t = 0;
        int y0_t = 0;
        int x1_t = 0;
        int y1_t = 0;
#pragma omp for schedule(dynamic, 1)
        for (int xy_w = 0; xy_w < nx * ny; xy_w++)
        {

            int x_w = xy_w % nx + 1;
            int y_w = xy_w / nx + 1;
            double size_w = y_w * x_w;
            double size_w1 = 1 / size_w;
            double size_b1 = 1 / (size_p - size_w);

            for (int y0 = 0; y0 <= ny - y_w; y0++)
            {
                for (int x0 = 0; x0 <= nx - x_w; x0++)
                {
                    int y1 = y0 + y_w;
                    int x1 = x0 + x_w;
                    double4_t sum_in = data_v[y1][x1] - data_v[y1][x0] - data_v[y0][x1] + data_v[y0][x0];
                    double4_t sum_out = sum_p - sum_in;
                    double4_t sum1 = sum_in * sum_in * size_w1;
                    double4_t sum2 = sum_out * sum_out * size_b1;
                    double sum_u = sum1[0] + sum1[1] + sum1[2] + sum2[0] + sum2[1] + sum2[2];
                    if (sum_u > u_t)
                    {
                        u_t = sum_u;
                        y0_t = y0;
                        x0_t = x0;
                        y1_t = y1;
                        x1_t = x1;
                    }
                }
            }
        }

#pragma omp critical
        {
            if (u_t > u)
            {
                u = u_t;
                y0_r = y0_t;
                x0_r = x0_t;
                y1_r = y1_t;
                x1_r = x1_t;
            }
        }
    }

    double4_t inner_sum = data_v[y1_r][x1_r] - data_v[y1_r][x0_r] - data_v[y0_r][x1_r] + data_v[y0_r][x0_r];
    double4_t outer_sum = sum_p - inner_sum;
    int r_size = (y1_r - y0_r) * (x1_r - x0_r);
    double4_t inner_s{(double)r_size, (double)r_size, (double)r_size, (double)1};
    double4_t inner = inner_sum / inner_s;
    double4_t outer_s{(double)(size_p - r_size), (double)(size_p - r_size), (double)(size_p - r_size), (double)1};
    double4_t outer = outer_sum / outer_s;

    Result result{y0_r, x0_r, y1_r, x1_r, {(float)outer[0], (float)outer[1], (float)outer[2]}, {(float)inner[0], (float)inner[1], (float)inner[2]}};

    return result;
}
