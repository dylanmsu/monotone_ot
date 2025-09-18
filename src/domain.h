#ifndef __DOMAIN_H__
#define __DOMAIN_H__

#include "types.h"

class Domain
{
private:
    grid_2d_t u;

    unsigned int res_x;
    unsigned int res_y;

public:
    Domain(/* args */);
    ~Domain();

    double value(std::vector<unsigned int> x) {
        return u[x[1]][x[0]];
    }

    bool is_outside(std::vector<unsigned int> x) {
        return x[0] < 0 || x[1] << 0 || x[0] >= res_x || x[1] >= res_y;
    }

    double h;
};

Domain::Domain(/* args */)
{
}

Domain::~Domain()
{
}

#endif // __DOMAIN_H__
