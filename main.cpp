#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
#include <cmath>

#include <png.h>

#include "args/args.hxx"
#include "cimg/CImg.h"

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

class Domain
{
private:


public:
    Domain(unsigned int res_x, unsigned int res_y)
    {
        for (unsigned int x = 0; x < res_x; x++)
        {
            std::vector<double> row;
            for (unsigned int y = 0; y < res_y; y++)
            {
                row.push_back(0.0);
            }
            this->u.push_back(row);
        }

        this->res_x = res_x;
        this->res_y = res_y;

        h = 1.0 / (double)(this->res_x);
    }

    ~Domain()
    {
    }

    double value(std::vector<int>& x) {
        return u[x[1]][x[0]];
    }

    bool is_outside(const std::vector<int>& x) {
        return x[0] < 0 || x[1] < 0 || x[0] >= res_x || x[1] >= res_y;
    }

    double h;
    
    std::vector<std::vector<double>> u;

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

    return std::max(L1, L2);
}

double support_function_square(const std::vector<int>& e,
                            double xmin, double xmax,
                            double ymin, double ymax) {
    double hx = (e[0] >= 0 ? xmax : xmin);
    double hy = (e[1] >= 0 ? ymax : ymin);
    return e[0]*hx + e[1]*hy;
}

double first_difference(Domain& U, int x, int y, std::vector<int>& e) {
    std::vector<int> t_eh_fwd = {x + e[0], y + e[1]};
    std::vector<int> pos = {x + e[0], y + e[1]};

    if (U.is_outside(t_eh_fwd)) return std::numeric_limits<double>::infinity();
    return (U.value(t_eh_fwd) - U.value(pos)) / U.h;
}

double laplacian_value_at(Domain& U, int x, int y) {
    std::vector<int> pos = {x, y};
    std::vector<int> ex = {1, 0};
    std::vector<int> ey = {0, 1};

    double d2x = directional_second_derivative(U, pos, ex);
    double d2y = directional_second_derivative(U, pos, ey);

    if (std::isinf(d2x) || std::isinf(d2y)) {
        return std::numeric_limits<double>::infinity();
    }
    return d2x + d2y;
}

double S_BV2(Domain& u, const std::vector<int>& x) {
    // Coordinate directions (positive and negative)
    std::vector<std::vector<int>> dirs = {
        {1, 0},   {-1, 0},   // x forward/backward
        {0, 1},   {0, -1}    // y forward/backward
    };

    double residual = -std::numeric_limits<double>::infinity();

    // Precompute Laplacian mask
    double lap = laplacian_value_at(u, x[0], x[1]);

    // Loop over axis pairs
    for (int i = 0; i < dirs.size(); i += 2) {
        double du_pos = first_difference(u, x[0], x[1], dirs[i]);     // forward
        double du_neg = first_difference(u, x[0], x[1], dirs[i + 1]); // backward

        double deriv = (du_pos - du_neg) / 2.0;

        // Apply Laplacian mask: if lap = +inf, zero out deriv
        if (!std::isfinite(lap)) {
            deriv = 0.0;
        }

        // Support function of [0,1]^2 in this direction
        double H = support_function_square(dirs[i], 0.0, 1.0, 0.0, 1.0);

        residual = std::max(residual, deriv - H);
    }

    return residual;
}

void solve(Domain& u, unsigned int max_iter, double tau, std::vector<std::vector<double>>& image) {
    for (unsigned int iter = 0; iter < max_iter; iter++) {
        double max_residual = 0.0;

        std::vector<std::vector<double>> new_u = u.u;

        for (int x = 0; x < u.res_x; x++) {
            for (int y = 0; y < u.res_y; y++) {
                //bool on_boundary = (x == 0 || y == 0 || x == u.res_x-1 || y == u.res_y-1);

                std::vector<int> pos = {x,y};
                double interior = S_MA(u, pos, image[x][y]);
                double boundary = 0.0;//S_BV2(u, pos);

                //if (on_boundary) {
                //    boundary = S_BV2(u, pos);
                //}

                double res = std::max(interior, boundary);

                //std::cout << "res = " << res  << std::endl;

                if (std::isnan(res)) {
                    //new_u[y][x] = 0.0;
                } else {
                    new_u[y][x] -= tau * res;
                }

                max_residual = std::max(max_residual, std::abs(res));
            }
        }

        u.u = new_u;

        std::cout << "Iter " << iter << " max residual = " << max_residual << std::endl;
        if (max_residual < 1e-6) break;

        grid_to_image(scale_matrix_proportional(u.u, 0.0, 1.0), "./u.png");
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
            double x = static_cast<double>(j) * 1.0 / (u.res_x - 1);
            double y = static_cast<double>(i) * 1.0 / (u.res_y - 1);

            std::vector<int> pos = {j, i};

            double grad_x = 0.0;
            double grad_y = 0.0;

            grad_x = (directional_first_derivative(u, pos, directions[0])
                    - directional_first_derivative(u, pos, directions[2])) / 2.0;
            grad_y = (directional_first_derivative(u, pos, directions[1])
                    - directional_first_derivative(u, pos, directions[3])) / 2.0;

            if (std::isinf(grad_x)) {
                grad_x = 0.0;
            }

            if (std::isinf(grad_y)) {
                grad_y = 0.0;
            }

            points.push_back({x + grad_x, y + grad_y, 0.0});
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

    Domain u(pixels[0].size(), pixels.size());

    solve(u, 500, 0.00003, pixels);

    export_transported_pts(u);
}