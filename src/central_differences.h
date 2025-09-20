#ifndef __CENTRAL_DIFFERENCES__
#define __CENTRAL_DIFFERENCES__

#include "domain.h"

#include <limits>

class central_differences
{
private:
    /* data */
public:
    central_differences(/* args */);
    ~central_differences();

    double central_differences::directional_second_derivative(Domain& u, std::vector<unsigned int> x, std::vector<int> e) {
        std::vector<unsigned int> t_eh_fwd = {x[0] + e[0], x[1] + e[1]};
        std::vector<unsigned int> t_eh_bwd = {x[0] - e[0], x[1] - e[1]};

        if (u.is_outside(t_eh_fwd) || u.is_outside(t_eh_bwd))
        {
            return std::numeric_limits<double>::infinity();
        }
        else
        {
            return (u.value(t_eh_fwd) + u.value(t_eh_bwd) - 2.0 * u.value(x)) / (u.h * u.h);
        }
    }

    double central_differences::directional_first_derivative(Domain& u, std::vector<unsigned int> x, std::vector<int> e) {
        std::vector<unsigned int> t_eh_fwd = {x[0] + e[0], x[1] + e[1]};

        if (u.is_outside(t_eh_fwd))
        {
            return std::numeric_limits<double>::infinity();
        }
        else
        {
            return (u.value(t_eh_fwd) - u.value(x)) / (u.h);
        }
    }

    double central_differences::closed_form_maximum(std::vector<int> v1, std::vector<int> v2, double m1, double m2, double b) {
        double norm1 = v1[0]*v1[0] + v1[1]*v1[1];
        double norm2 = v2[0]*v2[0] + v2[1]*v2[1];
        double term = (m1/(2.0*norm1) - m2/(2.0*norm2));
        double root = std::sqrt(b/(norm1*norm2) + term*term);
        return root - m1/(2.0*norm1) - m2/(2.0*norm2);
    }

    double central_differences::S_MA(Domain& u, std::vector<unsigned int> x, double b) {
        std::vector<double> derivatives(4);
        std::vector<std::vector<int>> directions = {
            {0, 1},
            {1, 0},
            {1, 1},
            {1, -1}
        };

        derivatives[0] = directional_second_derivative(u, x, directions[0]);
        derivatives[1] = directional_second_derivative(u, x, directions[1]);
        derivatives[2] = directional_second_derivative(u, x, directions[2]);
        derivatives[3] = directional_second_derivative(u, x, directions[3]);

        double L1 = closed_form_maximum(directions[1], directions[0], derivatives[1], derivatives[0], b);
        double L2 = closed_form_maximum(directions[2], directions[3], derivatives[2], derivatives[3], b);

        return std::max(L1, L2);
    }

    double support_function_square(const std::vector<int>& e,
                               double xmin, double xmax,
                               double ymin, double ymax) {
        double hx = (e[0] >= 0 ? xmax : xmin) * e[0];
        double hy = (e[1] >= 0 ? ymax : ymin) * e[1];
        return hx + hy;
    }

    double central_differences::S_BV2(Domain& u, std::vector<unsigned int> x) {
        std::vector<std::vector<int>> directions = {
            {0, 1},
            {1, 0},
            {1, 1},
            {1, -1}
        };

        double residual = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < directions.size(); i++)
        {
            double deriv = directional_first_derivative(u, x, directions[i]);
            double H = support_function_square(directions[i], 0.0, 1.0, 0.0, 1.0);
            residual = std::max(residual, deriv - H);
        }

        return residual;
    }
};

central_differences::central_differences(/* args */)
{
}

central_differences::~central_differences()
{
}


#endif //__CENTRAL_DIFFERENCES__