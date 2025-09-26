#ifndef __MONOTONE_RESIDUAL_H__
#define __MONOTONE_RESIDUAL_H__

#include "domain.h"

#include <numeric>
#include <limits>
#include <vector>
#include <cmath>

template <typename T>
class Monotone_residual
{
private:
    /* data */
public:
    Monotone_residual(/* args */)
    {
    }

    ~Monotone_residual()
    {
    }

    //*/

    /*T compute_residual(
        Domain<T>& u,
        const std::vector<std::vector<T>>& f_image,
        const std::vector<std::vector<T>>& g_image,
        const std::vector<std::vector<int>>& dirs,
        const std::vector<std::pair<int,int>>& bases,
        std::vector<std::vector<T>>& r_out)
    {
        T residual = 0.0;
        for (int x = 0; x < u.res_x; x++) {
            for (int y = 0; y < u.res_y; y++) {
                T element_area = T(1.0) / (u.res_x * u.res_y);
                std::vector<T> grad = u.gradient(y, x);
                T g = bilinearInterpolation(
                    g_image, 
                    (grad[0] + T(0.5)) * g_image.size(), 
                    (grad[1] + T(0.5)) * g_image.size()
                );
                T F = CppAD::CondExpLt(f_image[x][y] / g, T(1e3),
                                            f_image[x][y] / g, T(1e3))
                        * (3.141592 / 4);
                std::vector<int> pos = {x, y};
                T interior = S_MA(u, pos, F, dirs, bases);
                T boundary = S_BV2(u, pos, dirs) * T(50.0);
                T res = CppAD::CondExpGt(boundary, interior, boundary, interior);
                r_out[x][y] = res;
                residual += res * res;
            }
        }
        return residual / (u.res_x * u.res_y);
    }*/
};

#endif // __MONOTONE_RESIDUAL_H__
