#include <algorithm>
#include <random>
#include <omp.h>

typedef unsigned long long data_t;

data_t random(const int l, const int r)
{
    std::random_device rd;
    static thread_local std::mt19937 rng(rd());
    std::uniform_int_distribution<std::mt19937::result_type> dist(l, r - 1);
    return dist(rng);
}

void quicksort(data_t *d, data_t *l, data_t *r)
{
    int thread_max = omp_get_max_threads() * 4;

    if (l >= r)
        return;
    else if (r - l < thread_max)
    {
        std::sort(l, r);
        return;
    }

    data_t p = *(d + random(l - d, r - d));
    data_t *m1 = std::partition(l, r, [p](const auto &n)
                                { return n < p; });
    data_t *m2 = std::partition(m1, r, [p](const auto &n)
                                { return !(p < n); });

#pragma omp task
    quicksort(d, l, m1);

#pragma omp task
    quicksort(d, m2, r);
}

void psort(int n, data_t *d)
{
#pragma omp parallel
#pragma omp single nowait
    {
        quicksort(d, d, d + n);
    }
}
