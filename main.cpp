#include <fstream>
#include <iostream>
#include <numeric>
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
            double gray = image_grid[x][y];
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

void image_to_grid(const std::string& filename, std::vector<std::vector<double>>& image_grid) {
    FILE* fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        throw std::runtime_error("Failed to open PNG file.");
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        throw std::runtime_error("Failed to create PNG read struct.");
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(fp);
        throw std::runtime_error("Failed to create PNG info struct.");
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        throw std::runtime_error("Error during PNG read initialization.");
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16) {
        png_set_strip_16(png);
    }

    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png);
    }

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png);
    }

    if (png_get_valid(png, info, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png);
    }

    png_set_expand(png);  // Expand palettes, grayscale, etc.
    png_set_gray_to_rgb(png); // If grayscale, convert to RGB
    png_set_add_alpha(png, 0xFF, PNG_FILLER_AFTER); // Only add alpha if missing


    png_read_update_info(png, info);

    std::vector<png_bytep> row_pointers(height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_bytep)malloc(png_get_rowbytes(png, info));
    }

    png_read_image(png, row_pointers.data());

    fclose(fp);

    for (int i = 0; i < height; ++i) {
        std::vector<double> row;
        for (int j = 0; j < width; ++j) {
            png_bytep px = &row_pointers[i][j * 4];
            double r = px[0] / 255.0;
            double g = px[1] / 255.0;
            double b = px[2] / 255.0;
            double gray = (0.299 * r) + (0.587 * g) + (0.114 * b);
            row.push_back(gray);
        }
        image_grid.push_back(row);
    }

    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }

    png_destroy_read_struct(&png, &info, NULL);
}

void normalizeImage(std::vector<std::vector<double>> &F) {
    double sum = 0.0;

    // First, compute the total sum (integral)
    for (const auto &row : F) {
        for (double val : row) {
            sum += val;
        }
    }

    std::cout << "image integral = " << sum << std::endl;

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

void clamp(int &value, int min, int max) {
    value = std::max(std::min(value, max), min);
}

// Bilinear interpolation function
double bilinearInterpolation(const std::vector<std::vector<double>>& image, double x, double y) {
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
        printf("interpolation out of range: x: %f, y: %f\r\n", x, y);

        printf("x0: %i, y0: %i, x1: %i, y1: %i\r\n", x0, y0, x1, y1);
        // Handle out-of-bounds condition
        return 0.0;  // Default value
    }

    // Interpolate along x-axis
    double fx1 = x - x0;
    double fx0 = 1.0 - fx1;

    // Interpolate along y-axis
    double fy1 = y - y0;
    double fy0 = 1.0 - fy1;

    // Perform bilinear interpolation
    double top = fx0 * image[y0][x0] + fx1 * image[y0][x1];
    double bottom = fx0 * image[y1][x0] + fx1 * image[y1][x1];
    return std::max(fy0 * top + fy1 * bottom, 1e-12);
}

std::vector<std::vector<double>> hammingResize(
    const std::vector<std::vector<double>>& input_image,
    int current_resolution,
    int target_resolution)
{
    // Validate inputs
    if (current_resolution <= 0 || target_resolution <= 0) {
        return std::vector<std::vector<double>>(target_resolution,
                                                std::vector<double>(target_resolution, 0.0));
    }

    if (input_image.size() != current_resolution ||
        input_image[0].size() != current_resolution) {
        return std::vector<std::vector<double>>(target_resolution,
                                                std::vector<double>(target_resolution, 0.0));
    }

    if (current_resolution == target_resolution) {
        return input_image;
    }

    std::vector<std::vector<double>> output_image(target_resolution,
                                                  std::vector<double>(target_resolution, 0.0));

    double scale = static_cast<double>(current_resolution) / static_cast<double>(target_resolution);
    bool is_downscaling = scale > 1.0;
    double filter_scale = is_downscaling ? scale : 1.0;
    double support = 3.0 * filter_scale;

    // Hamming windowed sinc kernel
    auto hammingKernel = [filter_scale](double x) -> double {
        x = std::abs(x);
        double support = 3.0 * filter_scale;

        if (x >= support) return 0.0;
        if (x == 0.0) return 1.0;

        double norm_x = x / filter_scale;
        double pi_x = M_PI * norm_x;
        double sinc = std::sin(pi_x) / pi_x;

        double window_arg = M_PI * x / support;
        double hamming_window = 0.54 + 0.46 * std::cos(window_arg);

        return sinc * hamming_window;
    };

    // Safe pixel access
    auto getPixel = [&](int x, int y) -> double {
        x = std::max(0, std::min(x, current_resolution - 1));
        y = std::max(0, std::min(y, current_resolution - 1));
        return input_image[y][x];
    };

    for (int target_y = 0; target_y < target_resolution; ++target_y) {
        for (int target_x = 0; target_x < target_resolution; ++target_x) {
            double src_x = (target_x + 0.5) * scale - 0.5;
            double src_y = (target_y + 0.5) * scale - 0.5;

            double sum = 0.0;
            double weight_sum = 0.0;

            int x_start = static_cast<int>(std::floor(src_x - support));
            int x_end = static_cast<int>(std::ceil(src_x + support));
            int y_start = static_cast<int>(std::floor(src_y - support));
            int y_end = static_cast<int>(std::ceil(src_y + support));

            for (int sample_y = y_start; sample_y <= y_end; ++sample_y) {
                for (int sample_x = x_start; sample_x <= x_end; ++sample_x) {
                    double dx = src_x - sample_x;
                    double dy = src_y - sample_y;

                    double weight_x = hammingKernel(dx);
                    double weight_y = hammingKernel(dy);
                    double weight = weight_x * weight_y;

                    if (std::abs(weight) > 1e-10) {
                        sum += getPixel(sample_x, sample_y) * weight;
                        weight_sum += weight;
                    }
                }
            }

            output_image[target_y][target_x] = (weight_sum > 0.0) ? sum / weight_sum : 0.0;
        }
    }

    return output_image;
}

class Domain {
public:
    Domain(unsigned int res_x, unsigned int res_y)
    : res_x(res_x), res_y(res_y)
    {
        // store as u[row][col] == u[x][y]
        u.assign(res_y, std::vector<double>(res_x, 0.0));

        // if domain mapped to [0,1] in x, spacing is 1/(N-1)
        // if you map to [-1,1] use 2.0/(res_x-1)
        h = 1.0 / double(res_x - 1);
    }

    double value(const std::vector<int>& pos) const {
        int x = pos[0], y = pos[1];
        return u[x][y];
    }

    bool is_outside(const std::vector<int>& pos) const {
        int x = pos[0], y = pos[1];
        return x < 0 || y < 0 || x >= int(res_x) || y >= int(res_y);
    }

    double h;
    std::vector<std::vector<double>> u; // u[x][y]
    unsigned int res_x;
    unsigned int res_y;
};

class MA_Residual
{
private:
    Domain u;
public:
    MA_Residual(unsigned int resolution_x, unsigned int resolution_y);
    ~MA_Residual();
};

MA_Residual::MA_Residual(unsigned int resolution_x, unsigned int resolution_y)
    :u(resolution_x, resolution_y)
{
}

MA_Residual::~MA_Residual()
{
}


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
        return ((u.value(t_eh_fwd) - u.value(x)) / (u.h));
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

    return (grad0 + grad1) / std::sqrt(e[0]*e[0] + e[1]*e[1]);
}

double closed_form_maximum(std::vector<int>& v1, std::vector<int>& v2, double m1, double m2, double b) {
    double norm1 = v1[0]*v1[0] + v1[1]*v1[1];
    double norm2 = v2[0]*v2[0] + v2[1]*v2[1];
    double term = (m1/(2.0*norm1) - m2/(2.0*norm2));
    double root = std::sqrt(b/(norm1*norm2) + term*term);
    return root - m1/(2.0*norm1) - m2/(2.0*norm2);
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

double S_MA(Domain& u, const std::vector<int>& x, double b, const std::vector<std::vector<int>>& dirs, const std::vector<std::pair<int,int>>& bases) {
    std::vector<double> derivatives(dirs.size());
    for (size_t k = 0; k < dirs.size(); ++k) {
        derivatives[k] = directional_second_derivative(u, const_cast<std::vector<int>&>(x), 
                                                       const_cast<std::vector<int>&>(dirs[k]));
    }

    double residual = -std::numeric_limits<double>::infinity();

    for (auto [i,j] : bases) {
        double val = closed_form_maximum(const_cast<std::vector<int>&>(dirs[i]),
                                         const_cast<std::vector<int>&>(dirs[j]),
                                         derivatives[i], derivatives[j], b);
        residual = std::max(residual, val);
    }

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

double S_BV2(Domain& U, std::vector<int>& pos, const std::vector<std::vector<int>>& dirs) {
    double residue = -std::numeric_limits<double>::infinity();
    for (auto& e : dirs) {
        double De = directional_upwind_gradient(U, pos, const_cast<std::vector<int>&>(e));
        double sigma = 0.5;
        residue = std::max(residue, De - sigma);
    }
    return residue;
}

double compute_residual(Domain& u,
                        const std::vector<std::vector<double>>& image,
                        const std::vector<std::vector<int>>& dirs,
                        const std::vector<std::pair<int,int>>& bases,
                        std::vector<std::vector<double>>& r_out)
{
    double residual = 0.0;
    for (int x = 0; x < u.res_x; x++) {
        for (int y = 0; y < u.res_y; y++) {
            std::vector<int> pos = {x, y};
            double area_ratio = (std::sqrt(2) - 1) * 2.0;
            //double area_ratio = 3.141592 / 4;
            //double area_ratio = 0.5;
            double interior = S_MA(u, pos, image[x][y] * u.res_x * u.res_y * area_ratio, dirs, bases);
            double boundary = S_BV2(u, pos, dirs) * 20.0;
            double res = std::max(boundary, interior);
            r_out[x][y] = res;
            residual += res * res;
        }
    }

    return residual / u.res_x * u.res_y;
}

void export_transported_pts(Domain& u) {
    std::vector<std::vector<double>> points;

    // Generate points
    for (int i = 0; i < u.res_y; ++i) {
        for (int j = 0; j < u.res_x; ++j) {
            std::vector<int> pos = {j, i};

            double grad_x = 0.0;
            double grad_y = 0.0;

            // compute x gradient
            if (j > 0 && j < u.res_x-1) {
                grad_x = (u.u[i][j+1] - u.u[i][j-1]) / (2.0 * u.h);
            } else if (j == 0) {
                grad_x = (u.u[i][j+1] - u.u[i][j]) / u.h;
            } else {
                grad_x = (u.u[i][j] - u.u[i][j-1]) / u.h;
            }

            // compute y gradient
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

void solve(Domain& u, unsigned int max_iter, double tau, std::vector<std::vector<double>>& image) {
    std::vector<std::vector<double>> r = u.u;
    std::vector<std::vector<double>> d = u.u;

    const std::vector<std::vector<int>> dirs = generate_directions(3);
    const std::vector<std::pair<int,int>> bases = generate_bases(dirs);

    double prev_residual = 0.0;
    
    for (unsigned int iter = 0; iter < max_iter; iter++) {
        double residual = 0.0;

        std::vector<std::vector<double>> new_u = u.u;

        residual = compute_residual(u, image, dirs, bases, r);
        
        subtractAverage(r);
        poisson_solver(r, d, u.res_x, u.res_y, 1e6, 1e-6, 16);

        for (int x = 0; x < u.res_x; x++) {
            for (int y = 0; y < u.res_y; y++) {
                new_u[x][y] += tau * d[x][y];
            }
        }

        u.u = new_u;

        std::cout << "Iter " << iter << " max residual = " << residual << ", change = " << prev_residual - residual << std::endl;
        if (std::abs(prev_residual - residual) < 0.001) break;

        if (iter % 10 == 0) {
            export_transported_pts(u);
        }

        prev_residual = residual;
    }
}
//*/

int main(int argc, char** argv) {
    std::vector<std::vector<double>> S_MABV2;

    // Parse user arguments
    args::ArgumentParser parser("", "");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

    args::ValueFlag<std::string>    input_png(parser, "image", "Image input", {"input_png"});
    args::ValueFlag<unsigned int>   resolution(parser, "resolution", "Resolution", {"res_w"});
    
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
    unsigned int res_w = 100;

    if (input_png) { 
        image_filename = args::get(input_png);
    }

    if (resolution) {
        res_w = args::get(resolution);
    }

	std::vector<std::vector<double>> pixels;
    image_to_grid(image_filename.c_str(), pixels);
    //pixels = scale_matrix_proportional(pixels, 0.0, 1.0);
    pixels = hammingResize(pixels, pixels[0].size(), res_w);

    normalizeImage(pixels);


    Domain u(pixels[0].size(), pixels.size());

    solve(u, 10000, 1.0 / (u.res_x * u.res_y), pixels);
}