#include <algorithm>
#include <omp.h>

typedef unsigned long long data_t;

static inline int last_pow2(int x)
{
    return x == 1 ? 1 : 1 << ((64 - 1) - __builtin_clzl(x - 1));
}

void psort(int n, data_t *data)
{
    int threads = last_pow2(omp_get_max_threads());

    if (threads < 2)
        return std::sort(data, data + n);

    int block = (n + threads - 1) / threads;

#pragma omp parallel for
    for (int i = 0; i <= threads; i++)
    {
        int index = i * block;
        int start = std::min(index, n);
        int end = std::min(index + block, n);
        std::sort(data + start, data + end);
    }

    // merge pairs of blocks
    threads /= 2;

    while (threads > 0)
    {
#pragma omp parallel for
        for (int i = 0; i < threads; i++)
        {
            int index = i * 2;

            int start = index * block;
            int middle = std::min(start + block, n);
            int end = std::min(start + 2 * block, n);

            std::inplace_merge(data + start, data + middle, data + end);
        }
        block *= 2;
        threads /= 2;
    }
}
