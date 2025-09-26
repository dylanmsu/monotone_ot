#ifndef __DOMAIN_H__
#define __DOMAIN_H__

#include <limits>
#include <vector>
#include <cmath>

#define INFINITY 1e100

template <typename T>
class Domain {
public:
    Domain(unsigned int res_x, unsigned int res_y)
    : res_x(res_x), res_y(res_y)
    {
        // store as u[row][col] == u[x][y]
        u.assign(res_y, std::vector<T>(res_x, T(0.0)));

        // if domain mapped to [0,1] in x, spacing is 1/(N-1)
        // if you map to [-1,1] use 2.0/(res_x-1)
        h = T(1.0) / T(res_x - 1);
    }

    T value(const std::vector<int>& pos) const {
        int x = pos[0], y = pos[1];
        return u[x][y];
    }

    bool is_outside(const std::vector<int>& pos) const {
        int x = pos[0], y = pos[1];
        return x < 0 || y < 0 || x >= int(res_x) || y >= int(res_y);
    }

    bool is_infinite(T value) {
        if (value >= T(INFINITY)) {
            return true;
        }

        if (value <= -T(INFINITY)) {
            return true;
        }

        return false;
    }

    T infinity() {
        return T(INFINITY);
    }

    T directional_second_derivative(std::vector<int>& x, std::vector<int>& e) {
        std::vector<int> t_eh_fwd = {x[0] + e[0], x[1] + e[1]};
        std::vector<int> t_eh_bwd = {x[0] - e[0], x[1] - e[1]};

        if (is_outside(t_eh_fwd) || is_outside(t_eh_bwd))
        {
            return T(INFINITY);
        }
        else
        {
            return (value(t_eh_fwd) + value(t_eh_bwd) - T(2.0) * value(x)) / (h * h);
        }
    }

    T directional_first_derivative(std::vector<int>& x, std::vector<int>& e) {
        std::vector<int> t_eh_fwd = {x[0] + e[0], x[1] + e[1]};

        if (is_outside(t_eh_fwd))
        {
            return T(INFINITY);
        }
        else
        {
            return ((value(t_eh_fwd) - value(x)) / (h));
        }
    }

    T dot_product(std::vector<int>& a, std::vector<int>& b) {
        if (a.size() != b.size()) {
            return T(0.0);
        }

        T dot = 0.0;
        for (int i = 0; i < a.size(); i++)
        {
            dot += a[i] * b[i];
        }
        
        return dot;
    }

    T directional_upwind_gradient(std::vector<int>& x, std::vector<int>& e) {
        // Unit basis directions
        std::vector<int> e0  = {1, 0};
        std::vector<int> e1  = {0, 1};
        std::vector<int> e0_inv  = {-1, 0};
        std::vector<int> e1_inv  = {0, -1};

        T e_dot_e0 = dot_product(e, e0);
        T e_dot_e1 = dot_product(e, e1);

        T grad0_fwd = directional_first_derivative(x, e0);
        T grad0_bwd = directional_first_derivative(x, e0_inv);

        T grad1_fwd = directional_first_derivative(x, e1);
        T grad1_bwd = directional_first_derivative(x, e1_inv);

        T grad0 = 0.0;
        if (e_dot_e0 < 0) {
            if (!is_infinite(grad0_fwd)) grad0 += e_dot_e0 * grad0_fwd;
        }
        if (e_dot_e0 > 0) {
            if (!is_infinite(grad0_bwd)) grad0 -= e_dot_e0 * grad0_bwd;
        }

        T grad1 = 0.0;
        if (e_dot_e1 < 0) {
            if (!is_infinite(grad1_fwd)) grad1 += e_dot_e1 * grad1_fwd;
        }
        if (e_dot_e1 > 0) {
            if (!is_infinite(grad1_bwd)) grad1 -= e_dot_e1 * grad1_bwd;
        }

        return (grad0 + grad1) / T(std::sqrt(e[0]*e[0] + e[1]*e[1]));
    }

    std::vector<T> gradient(int i, int j) {
        T grad_x = 0.0;
        T grad_y = 0.0;

        // compute x gradient
        if (j > 0 && j < res_x-1) {
            grad_x = (u[i][j+1] - u[i][j-1]) / T(2.0 * h);
        } else if (j == 0) {
            grad_x = (u[i][j+1] - u[i][j]) / T(h);
        } else {
            grad_x = (u[i][j] - u[i][j-1]) / T(h);
        }

        // compute y gradient
        if (i > 0 && i < res_y-1) {
            grad_y = (u[i+1][j] - u[i-1][j]) / T(2.0 * h);
        } else if (i == 0) {
            grad_y = (u[i+1][j] - u[i][j]) / T(h);
        } else {
            grad_y = (u[i][j] - u[i-1][j]) / T(h);
        }

        return {grad_x, grad_y};
    }

    T closed_form_maximum(std::vector<int>& v1, std::vector<int>& v2, T m1, T m2, T b) {
        T norm1 = T(v1[0]*v1[0] + v1[1]*v1[1]);
        T norm2 = T(v2[0]*v2[0] + v2[1]*v2[1]);
        T term = (m1/(2.0*norm1) - m2/(2.0*norm2));
        T root = std::sqrt(b/(norm1*norm2) + term*term);
        return root - m1/(2.0*norm1) - m2/(2.0*norm2);
    }

    T S_MA(const std::vector<int>& x, T b, const std::vector<std::vector<int>>& dirs, const std::vector<std::pair<int,int>>& bases) {
        std::vector<T> derivatives(dirs.size());
        for (size_t k = 0; k < dirs.size(); ++k) {
            derivatives[k] = directional_second_derivative(const_cast<std::vector<int>&>(x), 
                                                        const_cast<std::vector<int>&>(dirs[k]));
        }

        T residual = -infinity();

        for (auto [i,j] : bases) {
            if (is_infinite(derivatives[i]) || is_infinite(derivatives[j])) {
                continue;
            }
            
            T val = closed_form_maximum(const_cast<std::vector<int>&>(dirs[i]),
                                            const_cast<std::vector<int>&>(dirs[j]),
                                            derivatives[i], derivatives[j], b);
            residual = std::max(residual, val);
        }

        if (std::isnan(residual)) {
            residual = -infinity();
        }
        return residual;
    }

    T support_function_square(const std::vector<int>& e,
                                T xmin, T xmax,
                                T ymin, T ymax) {
        T hx = (e[0] >= 0 ? xmax : xmin);
        T hy = (e[1] >= 0 ? ymax : ymin);
        return e[0]*hx + e[1]*hy;
    }

    T S_BV2(std::vector<int>& pos, const std::vector<std::vector<int>>& dirs) {
        T residue = -infinity();
        for (auto& e : dirs) {
            T De = directional_upwind_gradient(pos, const_cast<std::vector<int>&>(e));
            //T sigma = 0.5;
            T sigma = support_function_square(e, -0.5, 0.5, -0.5, 0.5);
            residue = std::max(residue, De - sigma);
        }
        return residue;
    }

    std::vector<std::vector<int>> generate_directions(int radius) {
        std::vector<std::vector<int>> dirs;
        for (int i=-radius; i<=radius; ++i)
        {
            for (int j=-radius; j<=radius; ++j)
            {
                if (i==0 && j==0) {
                    continue;
                }

                // primitive directions only
                if (std::gcd(i,j) != 1) {
                    continue;
                }

                dirs.push_back({i,j});
            }
        }
        return dirs;
    }

    std::vector<std::pair<int,int>> generate_bases(const std::vector<std::vector<int>>& dirs) {
        std::vector<std::pair<int,int>> bases;
        for (size_t i=0; i<dirs.size(); ++i)
        {
            for (size_t j=i+1; j<dirs.size(); ++j)
            {
                int det = dirs[i][0]*dirs[j][1] - dirs[i][1]*dirs[j][0];

                if (std::abs(det) == 1) {
                    bases.emplace_back(i,j);
                }
            }
        }
        return bases;
    }

    void clamp(int &value, int min, int max) {
        value = std::max(std::min(value, max), min);
    }

    // Bilinear interpolation function
    T bilinearInterpolation(const std::vector<std::vector<T>>& image, T x, T y) {
        int x0 = floor(x);
        int y0 = floor(y);
        int x1 = ceil(x);
        int y1 = ceil(y);

        clamp(x0, 0, image[0].size() - 1);
        clamp(x1, 0, image[0].size() - 1);
        clamp(y0, 0, image.size() - 1);
        clamp(y1, 0, image.size() - 1);

        //if (x < 0 || y < 0 || x > image[0].size() || y > image.size()) {
        //    return 0.0;
        //}

        // Check if the point is outside the image bounds
        if (x0 < 0 || y0 < 0 || x1 >= image[0].size() || y1 >= image.size()) {
            //printf("interpolation out of range: x: %f, y: %f\r\n", x, y);
            //printf("x0: %i, y0: %i, x1: %i, y1: %i\r\n", x0, y0, x1, y1);
            // Handle out-of-bounds condition
            return 0.0;  // Default value
        }

        // Interpolate along x-axis
        T fx1 = x - x0;
        T fx0 = 1.0 - fx1;

        // Interpolate along y-axis
        T fy1 = y - y0;
        T fy0 = 1.0 - fy1;

        // Perform bilinear interpolation
        T top = fx0 * image[y0][x0] + fx1 * image[y0][x1];
        T bottom = fx0 * image[y1][x0] + fx1 * image[y1][x1];
        return std::max(fy0 * top + fy1 * bottom, 1e-6);
    }

    T compute_residual(const std::vector<std::vector<T>>& f_image, const std::vector<std::vector<T>>& g_image, const std::vector<std::vector<int>>& dirs, const std::vector<std::pair<int,int>>& bases, std::vector<std::vector<T>>& r_out)
    {
        T residual = 0.0;
        for (int x = 0; x < res_x; x++) {
            for (int y = 0; y < res_y; y++) {

                T element_area = 1.0;

                std::vector<T> grad = gradient(y, x);
                
                //T area_ratio = (std::sqrt(2) - 1) * 2.0;
                //T area_ratio = T(3.141592 / 4.0);
                //T area_ratio = 0.5;
                T area_ratio = 1.0;

                //area_ratio += (2*u.res_x + 2*u.res_y) / (u.res_x * u.res_y);

                T g = bilinearInterpolation(g_image, (grad[0] + 0.5) * g_image.size(), (grad[1] + 0.5) * g_image.size());

                //T F = image[x][y] * u.res_x * u.res_y * area_ratio;

                T F = std::min(f_image[x][y] / element_area, 1e3) * area_ratio;

                std::vector<int> pos = {x, y};

                T interior = S_MA(pos, F, dirs, bases);
                T boundary = S_BV2(pos, dirs) * 50.0;
                T res = std::max(boundary, interior);
                r_out[x][y] = res;
                residual += res * res;
            }
        }

        return residual / res_x * res_y;
    }

    T h;
    std::vector<std::vector<T>> u; // u[x][y]
    unsigned int res_x;
    unsigned int res_y;
};

#endif // __DOMAIN_H__
