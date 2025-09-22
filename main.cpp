#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
#include <cmath>

#include <png.h>

#include "args/args.hxx"
#include "cimg/CImg.h"

#include "poisson_solver.h"

void export_grid_to_svg(std::vector<std::vector<double>> &points, double width, double height, int res_x, int res_y, std::string filename, double stroke_width) {
    std::ofstream svg_file(filename, std::ios::out);
    if (!svg_file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
    }

    // Write SVG header
    svg_file << "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n";
    svg_file << "<svg width=\"1000\" height=\"" << 1000.0f * (height / width) << "\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n";

    svg_file << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";

    for (int j = 0; j < res_y; j++) {
        std::string path_str = "M";
        for (int i = 0; i < res_x; i++) {
            int idx = i + j * res_x;

            const auto& point = points[idx];
            path_str += std::to_string((point[0] / width) * 1000.0f) + "," +
                        std::to_string((point[1] / height) * 1000.0f * (height / width));

            if (i < res_x - 1)
                path_str += "L";
        }
        svg_file << "<path d=\"" << path_str << "\" fill=\"none\" stroke=\"black\" stroke-width=\"" << stroke_width << "\"/>\n";
    }

    for (int j = 0; j < res_x; j++) {
        std::string path_str = "M";
        for (int i = 0; i < res_y; i++) {
            int idx = j + i * res_x;

            const auto& point = points[idx];
            path_str += std::to_string((point[0] / width) * 1000.0f) + "," +
                        std::to_string((point[1] / height) * 1000.0f * (height / width));

            if (i < res_x - 1)
                path_str += "L";
        }
        svg_file << "<path d=\"" << path_str << "\" fill=\"none\" stroke=\"black\" stroke-width=\"" << stroke_width << "\"/>\n";
    }

    // Write SVG footer
    svg_file << "</svg>\n";
    svg_file.close();
}

std::vector<std::vector<double>> scale_matrix_proportional(const std::vector<std::vector<double>>& matrix, double min_value, double max_value) {
    size_t rows = matrix.size();

    if (rows == 0) {
        throw std::invalid_argument("Input matrix is empty.");
    }

    size_t cols = matrix[0].size();

    // Check if all inner vectors have the same size
    for (size_t i = 1; i < rows; ++i) {
        if (matrix[i].size() != cols) {
            throw std::invalid_argument("Input matrix has inconsistent row sizes.");
        }
    }

    // Find the min and max values in the matrix
    double matrix_min = matrix[0][0];
    double matrix_max = matrix[0][0];

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (!std::isnan(matrix[i][j])) {
                matrix_min = std::min(matrix_min, matrix[i][j]);
                matrix_max = std::max(matrix_max, matrix[i][j]);
            }
        }
    }

    //std::cout << "maxtrix range = " << matrix_max - matrix_min << std::endl;

    // Perform proportional scaling
    std::vector<std::vector<double>> scaled_matrix(rows, std::vector<double>(cols));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (!std::isnan(matrix[i][j])) {
                if ((matrix_max - matrix_min) < 1e-12) {
                    scaled_matrix[i][j] = (min_value + max_value)/2;
                } else {
                    scaled_matrix[i][j] = min_value + (max_value - min_value) * (matrix[i][j] - matrix_min) / (matrix_max - matrix_min);
                }
            }
        }
    }

    return scaled_matrix;
}

void image_to_grid(const cimg_library::CImg<unsigned char>& image, std::vector<std::vector<double>>& image_grid) {
    for (int i = 0; i < image.height(); ++i) {
        std::vector<double> row;
        for (int j = 0; j < image.width(); ++j) {
			double r = image(i, j, 0) / 255.0; // Normalize R component
			double g = image(i, j, 1) / 255.0; // Normalize G component
			double b = image(i, j, 2) / 255.0; // Normalize B component
			double value = (0.299 * r) + (0.587 * g) + (0.114 * b); // Calculate grayscale value using luminosity method
            row.push_back(value);
        }
        image_grid.push_back(row);
    }
}

void grid_to_image(const std::vector<std::vector<double>>& image_grid, const std::string& filename) {
    if (image_grid.empty()) {
        throw std::runtime_error("Image grid is empty.");
    }
    size_t height = image_grid.size();
    size_t width = image_grid[0].size();
    if (width == 0 || height == 0) {
        throw std::runtime_error("Image grid has invalid dimensions (zero width or height).");
    }
    for (const auto& row : image_grid) {
        if (row.size() != width) {
            throw std::runtime_error("Image grid rows have inconsistent lengths.");
        }
    }

    FILE* fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        throw std::runtime_error("Failed to open file for writing.");
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        fclose(fp);
        throw std::runtime_error("Failed to create PNG write struct.");
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, nullptr);
        fclose(fp);
        throw std::runtime_error("Failed to create PNG info struct.");
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        throw std::runtime_error("Error during PNG creation.");
    }

    png_init_io(png, fp);

    png_set_IHDR(
        png,
        info,
        width,
        height,
        8,
        PNG_COLOR_TYPE_RGBA,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );

    png_write_info(png, info);

    std::vector<png_bytep> row_pointers(height);
    for (size_t y = 0; y < height; ++y) {
        row_pointers[y] = static_cast<png_bytep>(png_malloc(png, png_get_rowbytes(png, info)));
        for (size_t x = 0; x < width; ++x) {
            double gray = image_grid[y][x];
            // Clamp the grayscale value to [0.0, 1.0]
            gray = std::max(0.0, std::min(gray, 1.0));
            png_byte value = static_cast<png_byte>(gray * 255.0);
            png_bytep pixel = &(row_pointers[y][x * 4]);
            pixel[0] = value; // Red
            pixel[1] = value; // Green
            pixel[2] = value; // Blue
            pixel[3] = 255;   // Alpha (fully opaque)
        }
    }

    png_write_image(png, row_pointers.data());
    png_write_end(png, nullptr);

    // Cleanup
    for (size_t y = 0; y < height; ++y) {
        png_free(png, row_pointers[y]);
    }
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

void normalizeImage(std::vector<std::vector<double>> &F) {
    double sum = 0.0;

    // First, compute the total sum (integral)
    for (const auto &row : F) {
        for (double val : row) {
            sum += val;
        }
    }

    // Avoid division by zero
    if (sum == 0.0) {
        std::cerr << "Error: Integral (sum) is zero, cannot normalize." << std::endl;
        return;
    }

    // Normalize each value
    for (auto &row : F) {
        for (double &val : row) {
            val /= sum;
        }
    }
}

void subtractAverage(std::vector<std::vector<double>>& raster) {
    // Calculate the average of the raster
    double sum = 0.0;
    int count = 0;

    for (const auto& row : raster) {
        for (double value : row) {
            if (!std::isnan(value)) {
                sum += value;
                count++;
            }
        }
    }

    double average = sum / count;

    // Subtract the average from each element of the raster
    for (auto& row : raster) {
        std::transform(row.begin(), row.end(), row.begin(), [average](double value) {
            if (std::isnan(value)) {
                return value;
            } else {
                return value - average;
            }
        });
    }
}

class Domain {
public:
    Domain(unsigned int res_x, unsigned int res_y)
    : res_x(res_x), res_y(res_y)
    {
        // store as u[row][col] == u[y][x]
        u.assign(res_y, std::vector<double>(res_x, 0.0));

        // if domain mapped to [0,1] in x, spacing is 1/(N-1)
        // if you map to [-1,1] use 2.0/(res_x-1)
        h = 1.0 / double(res_x - 1);
    }

    double value(const std::vector<int>& pos) const {
        int x = pos[0], y = pos[1];
        return u[y][x];
    }

    bool is_outside(const std::vector<int>& pos) const {
        int x = pos[0], y = pos[1];
        return x < 0 || y < 0 || x >= int(res_x) || y >= int(res_y);
    }

    double h;
    std::vector<std::vector<double>> u; // u[y][x]
    unsigned int res_x;
    unsigned int res_y;
};

double directional_second_derivative(Domain& u, std::vector<int>& x, std::vector<int>& e) {
    std::vector<int> t_eh_fwd = {x[0] + e[0], x[1] + e[1]};
    std::vector<int> t_eh_bwd = {x[0] - e[0], x[1] - e[1]};

    if (u.is_outside(t_eh_fwd) || u.is_outside(t_eh_bwd))
    {
        return std::numeric_limits<double>::infinity();
    }
    else
    {
        return (u.value(t_eh_fwd) + u.value(t_eh_bwd) - 2.0 * u.value(x)) / (u.h * u.h);
    }
}

double directional_first_derivative(Domain& u, std::vector<int>& x, std::vector<int>& e) {
    std::vector<int> t_eh_fwd = {x[0] + e[0], x[1] + e[1]};

    if (u.is_outside(t_eh_fwd))
    {
        return std::numeric_limits<double>::infinity();
    }
    else
    {
        return (u.value(t_eh_fwd) - u.value(x)) / (u.h);
    }
}

double dot_product(std::vector<int>& a, std::vector<int>& b) {
    if (a.size() != b.size()) {
        return 0.0;
    }

    double dot = 0.0;
    for (int i = 0; i < a.size(); i++)
    {
        dot += a[i] * b[i];
    }
    
    return dot;
}

double directional_upwind_gradient(Domain& u, std::vector<int>& x, std::vector<int>& e) {
    // Unit basis directions
    std::vector<int> e0  = {1, 0};
    std::vector<int> e1  = {0, 1};
    std::vector<int> e0_inv  = {-1, 0};
    std::vector<int> e1_inv  = {0, -1};

    double e_dot_e0 = dot_product(e, e0);
    double e_dot_e1 = dot_product(e, e1);

    double grad0_fwd = directional_first_derivative(u, x, e0);
    double grad0_bwd = directional_first_derivative(u, x, e0_inv);

    double grad1_fwd = directional_first_derivative(u, x, e1);
    double grad1_bwd = directional_first_derivative(u, x, e1_inv);

    double grad0 = 0.0;
    if (e_dot_e0 < 0) {
        if (!std::isinf(grad0_fwd)) grad0 += e_dot_e0 * grad0_fwd;
    }
    if (e_dot_e0 > 0) {
        if (!std::isinf(grad0_bwd)) grad0 -= e_dot_e0 * grad0_bwd;
    }

    double grad1 = 0.0;
    if (e_dot_e1 < 0) {
        if (!std::isinf(grad1_fwd)) grad1 += e_dot_e1 * grad1_fwd;
    }
    if (e_dot_e1 > 0) {
        if (!std::isinf(grad1_bwd)) grad1 -= e_dot_e1 * grad1_bwd;
    }

    return grad0 + grad1;
}

double closed_form_maximum(std::vector<int>& v1, std::vector<int>& v2, double m1, double m2, double b) {
    double norm1 = v1[0]*v1[0] + v1[1]*v1[1];
    double norm2 = v2[0]*v2[0] + v2[1]*v2[1];
    double term = (m1/(2.0*norm1) - m2/(2.0*norm2));
    double root = std::sqrt(b/(norm1*norm2) + term*term);
    return root - m1/(2.0*norm1) - m2/(2.0*norm2);
}

double S_MA(Domain& u, std::vector<int>& x, double b) {
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

    //std::cout << "derivatives[0] = " << derivatives[0] << std::endl;
    //std::cout << "derivatives[1] = " << derivatives[1] << std::endl;
    //std::cout << "derivatives[2] = " << derivatives[2] << std::endl;
    //std::cout << "derivatives[3] = " << derivatives[3] << std::endl;

    double L1 = closed_form_maximum(directions[1], directions[0], derivatives[1], derivatives[0], b);
    double L2 = closed_form_maximum(directions[2], directions[3], derivatives[2], derivatives[3], b);

    //std::cout << "L1, L2 = " << L1 << ", " << L2 << std::endl;

    double residual = std::max(L1, L2);

    if (std::isnan(residual)) {
        residual = -std::numeric_limits<double>::infinity();
    }

    return residual;
}

double support_function_square(const std::vector<int>& e,
                            double xmin, double xmax,
                            double ymin, double ymax) {
    double hx = (e[0] >= 0 ? xmax : xmin);
    double hy = (e[1] >= 0 ? ymax : ymin);
    return e[0]*hx + e[1]*hy;
}

double S_BV2(Domain& U, std::vector<int>& pos) {
    // sample directions with Euclidean norm 1: axis and diagonals
    std::vector<std::vector<int>> sample_dirs = {
        {1,0}, {-1,0}, {0,1}, {0,-1},
    };
    double residue = -std::numeric_limits<double>::infinity();
    for (auto& e : sample_dirs) {
        // normalize e length for sigma: but upwind_D_e uses integer e; here we treat sigma separately
        double De = directional_upwind_gradient(U, pos, e);
        double sigma = support_function_square(e, -0.5, 0.5, -0.5, 0.5); // if square target, compute differently
        //sigma = 0.5;
        residue = std::max(residue, De - sigma);
    }
    return residue;
}

void solve(Domain& u, unsigned int max_iter, double tau, std::vector<std::vector<double>>& image) {
    for (unsigned int iter = 0; iter < max_iter; iter++) {
        double max_residual = 0.0;

        std::vector<std::vector<double>> new_u = u.u;

        std::vector<std::vector<double>> r = u.u;
        std::vector<std::vector<double>> d = u.u;

        for (int x = 0; x < u.res_x; x++) {
            for (int y = 0; y < u.res_y; y++) {
                bool on_boundary = (x == 0 || y == 0 || x == u.res_x-1 || y == u.res_y-1);

                std::vector<int> pos = {x,y};
                double interior = S_MA(u, pos, image[x][y] / (u.h * u.h));
                double boundary = S_BV2(u, pos);

                double res = std::max(boundary, interior);

                //std::cout << "res = " << res  << std::endl;

                //new_u[y][x] -= tau * res;

                r[y][x] = res;

                max_residual = std::max(max_residual, std::abs(res));
            }
        }

        
        subtractAverage(r);
        poisson_solver(r, d, u.res_x, u.res_y, 1e6, 1e-6, 16);

        for (int x = 0; x < u.res_x; x++) {
            for (int y = 0; y < u.res_y; y++) {
                new_u[y][x] += tau * d[y][x];
            }
        }
        //*/

        u.u = new_u;

        std::cout << "Iter " << iter << " max residual = " << max_residual << std::endl;
        if (max_residual < 1e-6) break;

        //grid_to_image(scale_matrix_proportional(r, 0.0, 1.0), "./r.png");
    }
}

void export_transported_pts(Domain& u) {
    std::vector<std::vector<double>> points;

    std::vector<std::vector<int>> directions = {
        {1, 0},
        {0, 1},
        {-1, 0},
        {0, -1},
    };

    // Generate points
    for (int i = 0; i < u.res_y; ++i) {
        for (int j = 0; j < u.res_x; ++j) {
            double x = static_cast<double>(j) * 1.0 / (u.res_x);
            double y = static_cast<double>(i) * 1.0 / (u.res_y);

            std::vector<int> pos = {j, i};

            double grad_x = 0.0;
            double grad_y = 0.0;

            // asymetrical but seems correct (not actually sure) on the +x +y boundaries. the -x, -y boundaries lie on the origin,
            //grad_x = directional_upwind_gradient(u, pos, directions[0]);
            //grad_y = directional_upwind_gradient(u, pos, directions[1]);

            // all four boundaries seem to show the same problem now. the boundary is exactly halfway between the origin and where it should be
            //grad_x = (directional_upwind_gradient(u, pos, directions[0]) - directional_upwind_gradient(u, pos, directions[2])) / 2.0;
            //grad_y = (directional_upwind_gradient(u, pos, directions[1]) - directional_upwind_gradient(u, pos, directions[3])) / 2.0;

            if (j > 0 && j < u.res_x-1) {
                grad_x = (u.u[i][j+1] - u.u[i][j-1]) / (2.0 * u.h);
            } else if (j == 0) {
                grad_x = (u.u[i][j+1] - u.u[i][j]) / u.h;
            } else {
                grad_x = (u.u[i][j] - u.u[i][j-1]) / u.h;
            }

            if (i > 0 && i < u.res_y-1) {
                grad_y = (u.u[i+1][j] - u.u[i-1][j]) / (2.0 * u.h);
            } else if (i == 0) {
                grad_y = (u.u[i+1][j] - u.u[i][j]) / u.h;
            } else {
                grad_y = (u.u[i][j] - u.u[i-1][j]) / u.h;
            }

            if (std::isinf(grad_x)) {
                grad_x = 0.0;
            }

            if (std::isinf(grad_y)) {
                grad_y = 0.0;
            }

            points.push_back({grad_x+0.5, grad_y+0.5, 0.0});
        }
    }

    // export transported grid
    export_grid_to_svg(points, 1.0, 1.0, u.res_x, u.res_y, "./grid.svg", 1.0);
}

int main(int argc, char** argv) {
    std::vector<std::vector<double>> S_MABV2;

    // Parse user arguments
    args::ArgumentParser parser("", "");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

    args::ValueFlag<std::string>    input_png(parser, "image", "Image input", {"input_png"});
    
    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (const args::Completion& e)
    {
        std::cout << e.what();
        return 0;
    }
    catch (const args::Help&)
    {
        std::cout << parser;
        return 0;
    }
    catch (const args::ParseError& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    // default values
    std::string image_filename = "";

    bool output_progress = false;

    if (input_png) { 
        image_filename = args::get(input_png);
    }

    cimg_library::CImg<unsigned char> image(image_filename.c_str());

	std::vector<std::vector<double>> pixels;
    image_to_grid(image, pixels);

    normalizeImage(pixels);

    Domain u(pixels[0].size(), pixels.size());

    for (int i = 0; i < 10000; i++)
    {
        solve(u, 10, 0.00002, pixels);
        export_transported_pts(u);
    }

}