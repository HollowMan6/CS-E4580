#include <algorithm>
#include <omp.h>

typedef unsigned long long data_t;

void psort(int n, data_t *data)
{
    int threads = omp_get_max_threads() * 99999;
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < n; i += threads)
        std::sort(data + i, data + std::min((i + threads), n));

    for (int i = threads; i < n; i *= 2)
#pragma omp parallel for schedule(static, 1)
        for (int j = 0; j < n; j = j + 2 * i)
            std::inplace_merge(data + j, data + std::min(j + i, n), data + std::min((j + 2 * i), n));
}
