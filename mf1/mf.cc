#include <algorithm>
#include <vector>
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/
void mf(int ny, int nx, int hy, int hx, const float *in, float *out)
{
  for (int y = 0; y < ny; y++)
  {
    for (int x = 0; x < nx; x++)
    {
      int xmin = std::max(x - hx, 0);
      int xmax = std::min(x + hx + 1, nx);
      int ymin = std::max(y - hy, 0);
      int ymax = std::min(y + hy + 1, ny);
      int size = (xmax - xmin) * (ymax - ymin);
      std::vector<double> values(size);
      int index = 0;
      for (int j = ymin; j < ymax; j++)
      {
        for (int i = xmin; i < xmax; i++)
        {
          values[index] = in[i + j * nx];
          index++;
        }
      }

      int middle = size / 2;
      std::nth_element(values.begin(), values.begin() + middle, values.end());
      if (size % 2 == 0)
      {
        std::nth_element(values.begin(), values.begin() + middle - 1, values.end());
        out[x + y * nx] = (values[middle] + values[middle - 1]) / 2.0;
      }
      else
      {
        out[x + y * nx] = values[middle];
      }
    }
  }
}
