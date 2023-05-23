#include <vector>
#include <x86intrin.h>

typedef float float8_t __attribute__((vector_size(8 * sizeof(float))));

float8_t zero{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

struct Result
{
  int y0;
  int x0;
  int y1;
  int x1;
  float out[3];
  float in[3];
};

Result segment(int ny, int nx, const float *data)
{
  std::vector<std::vector<float>> data_v(ny + 1, std::vector<float>(nx + 1));
#pragma omp parallel for schedule(dynamic, 1)
  for (int y = 1; y <= ny; y++)
  {
    for (int x = 1; x <= nx; x++)
    {
      data_v[y][x] = data[3 * (x - 1) + 3 * nx * (y - 1)] + data_v[y][x - 1];
    }
  }
#pragma omp parallel for schedule(dynamic, 1)
  for (int x = 1; x <= nx; x++)
  {
    for (int y = 1; y <= ny; y++)
    {
      data_v[y][x] += data_v[y - 1][x];
    }
  }

  int size_p = ny * nx;
  float sum = data_v[ny][nx];
  float8_t sum_float8 = {sum, sum, sum, sum,
                         sum, sum, sum, sum};

  int wy_r = 1;
  int wx_r = 1;

  float best_u = -1;
#pragma omp parallel for schedule(dynamic, 1)
  for (int xy_w = 0; xy_w < nx * ny; xy_w++)
  {
    int x_w = xy_w % nx + 1;
    int y_w = xy_w / nx + 1;
    float size_w = y_w * x_w;
    float size_wd = 1 / size_w;
    float size_bd = 1 / (size_p - size_w);

    float8_t max_float8 = zero;
    float best = -1;
    for (int y0 = 0; y0 <= ny - y_w; y0++)
    {
      int y1 = y0 + y_w;

      int x0_b = nx - x_w + 1;
      int blocks = x0_b / 8;
      for (int b = 0; b < blocks; b++)
      {
        int x0_start = 8 * b;
        int x1_start = x0_start + x_w;

        float8_t s1 = _mm256_loadu_ps(&data_v[y1][x1_start]);
        float8_t s2 = _mm256_loadu_ps(&data_v[y1][x0_start]);
        float8_t s3 = _mm256_loadu_ps(&data_v[y0][x1_start]);
        float8_t s4 = _mm256_loadu_ps(&data_v[y0][x0_start]);

        float8_t v_in = s1 - s2 - s3 + s4;
        float8_t v_out = sum_float8 - v_in;
        float8_t v = v_in * v_in * size_wd + v_out * v_out * size_bd;
        max_float8 = max_float8 > v ? max_float8 : v;
      }

      for (int x0 = 8 * blocks; x0 < x0_b; ++x0)
      {
        int x1 = x0 + x_w;

        float sum_in =
            data_v[y1][x1] - data_v[y1][x0] - data_v[y0][x1] + data_v[y0][x0];
        float sum_out = sum - sum_in;
        float util = sum_in * sum_in * size_wd + sum_out * sum_out * size_bd;
        if (util > best)
        {
          best = util;
        }
      }
    }

    float max_v = 0;
    for (int i = 0; i < 8; ++i)
    {
      if (max_float8[i] > max_v)
      {
        max_v = max_float8[i];
      }
    }
    if (max_v > best)
    {
      best = max_v;
    }

#pragma omp critical
    {
      if (best > best_u)
      {
        best_u = best;
        wx_r = x_w;
        wy_r = y_w;
      }
    }
  }

  int y0_r = 0;
  int x0_r = 0;
  int y1_r = 0;
  int x1_r = 0;
  float in = 0;
  float out = 0;

  float size_w = wx_r * wy_r;
  float size_wd = 1 / size_w;
  float size_bd = 1 / (size_p - size_w);
  best_u = -1;

#pragma omp parallel for schedule(dynamic, 1)
  for (int y0 = 0; y0 <= ny - wy_r; y0++)
  {
    for (int x0 = 0; x0 <= nx - wx_r; x0++)
    {
      int y1 = y0 + wy_r;
      int x1 = x0 + wx_r;

      float sum_in =
          data_v[y1][x1] - data_v[y1][x0] - data_v[y0][x1] + data_v[y0][x0];
      float sum_out = sum - sum_in;
      float util = sum_in * sum_in * size_wd + sum_out * sum_out * size_bd;

#pragma omp critical
      {
        if (util > best_u)
        {
          best_u = util;
          y0_r = y0;
          x0_r = x0;
          y1_r = y1;
          x1_r = x1;
          in = sum_in * size_wd;
          out = sum_out * size_bd;
        }
      }
    }
  }

  Result result{y0_r, x0_r, y1_r, x1_r, {out, out, out}, {in, in, in}};
  return result;
}
