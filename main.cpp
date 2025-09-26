#include <fstream>
#include <iostream>
#include <numeric>
#include <limits>
#include <deque>
#include <vector>
#include <cmath>

#include <png.h>

#include "cppad/cppad.hpp"
#include "args/args.hxx"
#include "cimg/CImg.h"

#include "mesh.h"
#include "poisson_solver.h"

#include "monge_ampere/domain.h"
#include "monge_ampere/monge_ampere_solver.h"
#include "normal_integration/normal_integration.h"

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

std::vector<std::vector<double>> apply_inverse_map(
    Mesh& tmap, 
    std::vector<std::vector<double>>& input_pts
) {

    std::vector<std::vector<double>> unit_points;

    for (int i = 0; i < tmap.res_y; ++i) {
        for (int j = 0; j < tmap.res_x; ++j) {
            double x = static_cast<double>(j) * 1.0 / (tmap.res_x - 1);
            double y = static_cast<double>(i) * 1.0 / (tmap.res_y - 1);
            unit_points.push_back({x, y, 0.0});
        }
    }
    
    // Build BVH over transported (target) points
    tmap.build_bvh(5, 30);

    std::vector<std::vector<double>> inverted_points;
    inverted_points.reserve(input_pts.size());

    for (size_t i = 0; i < input_pts.size(); ++i) {
        std::vector<Hit> hits;
        bool intersection = false;

        // Query input point against target BVH
        tmap.bvh->query(input_pts[i], hits, intersection);

        if (intersection && !hits.empty()) {
            const auto& tri = tmap.triangles[hits[0].face_id];

            // Get source triangle vertices
            const auto& v0 = unit_points[tri[0]];
            const auto& v1 = unit_points[tri[1]];
            const auto& v2 = unit_points[tri[2]];

            // Barycentric weights from target query
            double b0 = hits[0].barycentric_coords[0];
            double b1 = hits[0].barycentric_coords[1];
            double b2 = hits[0].barycentric_coords[2];

            // Interpolate in source domain
            double interpolation_x = v0[0]*b0 + v1[0]*b1 + v2[0]*b2;
            double interpolation_y = v0[1]*b0 + v1[1]*b1 + v2[1]*b2;

            inverted_points.push_back({interpolation_x, interpolation_y});
        } else {
            // No intersection → fallback: copy input
            inverted_points.push_back(input_pts[i]);
        }
    }

    return inverted_points;
}

void translatePoints(std::vector<std::vector<double>>& trg_pts, std::vector<double> position_xyz) {
  for (int i = 0; i < trg_pts.size(); i++)
  {
    trg_pts[i][0] += position_xyz[0];
    trg_pts[i][1] += position_xyz[1];
    trg_pts[i][2] += position_xyz[2];
  }
}

void normalizeImage(std::vector<std::vector<double>> &F, double width, double height) {
    int Nx = F.size();                  // number of rows
    if (Nx == 0) return;
    int Ny = F[0].size();               // number of columns

    double dx = width / Ny;             // pixel width
    double dy = height / Nx;            // pixel height
    double dA = dx * dy;                // pixel area

    double integral = 0.0;

    // Compute integral using Riemann sum
    for (const auto &row : F) {
        for (double val : row) {
            integral += val * dA;
        }
    }

    std::cout << "image integral = " << integral << std::endl;

    if (integral == 0.0) {
        std::cerr << "Error: Integral is zero, cannot normalize." << std::endl;
        return;
    }

    // Normalize so the integral becomes 1
    for (auto &row : F) {
        for (double &val : row) {
            val /= integral;
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

std::vector<double> normalize_vec(std::vector<double> p1) {
    std::vector<double> vec(3);
    double squared_len = 0;
    for (int i=0; i<p1.size(); i++) {
        squared_len += p1[i] * p1[i];
    }

    double len = std::sqrt(squared_len);

    for (int i=0; i<p1.size(); i++) {
        vec[i] = p1[i] / len;
    }

    return vec;
}

//compute the desired normals
std::vector<std::vector<double>> fresnelMapping(
  std::vector<std::vector<double>> &vertices, 
  std::vector<std::vector<double>> &target_pts, 
  double refractive_index
) {
    std::vector<std::vector<double>> desiredNormals;

    //double boundary_z = -0.1;

    //vector<std::vector<double>> boundary_points;

    bool use_point_src = false;
    bool use_reflective_caustics = false;

    std::vector<double> pointLightPosition(3);
    pointLightPosition[0] = 0.5;
    pointLightPosition[1] = 0.5;
    pointLightPosition[2] = 0.5;

    // place initial points on the refractive surface where the light rays enter the material
    /*if (use_point_src && !use_reflective_caustics) {
        for(int i = 0; i < vertices.size(); i++) {
            std::vector<double> boundary_point(3);

            // ray to plane intersection to get the initial points
            double t = ((boundary_z - pointLightPosition[2]) / (vertices[i][2] - pointLightPosition[2]));
            boundary_point[0] = pointLightPosition[0] + t*(vertices[i][0] - pointLightPosition[0]);
            boundary_point[1] = pointLightPosition[1] + t*(vertices[i][1] - pointLightPosition[1]);
            boundary_point[2] = boundary_z;
            boundary_points.push_back(boundary_point);
        }
    }*/

    // run gradient descent on the boundary points to find their optimal positions such that they satisfy Fermat's principle
    /*if (!use_reflective_caustics && use_point_src) {
        for (int i=0; i<boundary_points.size(); i++) {
            for (int iteration=0; iteration<100000; iteration++) {
                double grad_x;
                double grad_y;
                gradient(pointLightPosition, boundary_points[i], vertices[i], 1.0, refractive_index, grad_x, grad_y);

                boundary_points[i][0] -= 0.1 * grad_x;
                boundary_points[i][1] -= 0.1 * grad_y;

                // if magintude of both is low enough
                if (grad_x*grad_x + grad_y*grad_y < 0.000001) {
                    break;
                }
            }
        }
    }*/

    for(int i = 0; i < vertices.size(); i++) {
        std::vector<double> incidentLight(3);
        std::vector<double> transmitted = {
            target_pts[i][0] - vertices[i][0],
            target_pts[i][1] - vertices[i][1],
            target_pts[i][2] - vertices[i][2]
        };

        if (use_point_src) {
            incidentLight[0] = vertices[i][0] - pointLightPosition[0];
            incidentLight[1] = vertices[i][1] - pointLightPosition[1];
            incidentLight[2] = vertices[i][2] - pointLightPosition[2];
        } else {
            incidentLight[0] = 0;
            incidentLight[1] = 0;
            incidentLight[2] = -1;
        }

        transmitted = normalize_vec(transmitted);
        incidentLight = normalize_vec(incidentLight);

        std::vector<double> normal(3);
        if (use_reflective_caustics) {
            normal[0] = ((transmitted[0]) + incidentLight[0]) * 1.0f;
            normal[1] = ((transmitted[1]) + incidentLight[1]) * 1.0f;
            normal[2] = ((transmitted[2]) + incidentLight[2]) * 1.0f;
        } else {
            normal[0] = ((transmitted[0]) - (incidentLight[0]) * refractive_index) * -1.0f;
            normal[1] = ((transmitted[1]) - (incidentLight[1]) * refractive_index) * -1.0f;
            normal[2] = ((transmitted[2]) - (incidentLight[2]) * refractive_index) * -1.0f;
        }

        normal = normalize_vec(normal);

        desiredNormals.push_back(normal);
    }

    return desiredNormals;
}

Mesh extract_map(Domain<double>& u) {
    Mesh mesh(0.0, 1.0, 0.0, 1.0, u.res_x, u.res_y);

    // Generate points
    for (int i = 0; i < u.res_y; ++i) {
        for (int j = 0; j < u.res_x; ++j) {
            std::vector<int> pos = {j, i};

            std::vector<double> grad = u.gradient(i, j);

            int index = i * u.res_x + j;

            mesh.source_points[index] = {grad[0]+0.5, grad[1]+0.5, 0.0};
        }
    }

    return mesh;
}

void export_transported_pts(Domain<double>& u) {
    std::vector<std::vector<double>> points;

    Mesh tmap = extract_map(u);

    // export transported grid
    export_grid_to_svg(tmap.source_points, 1.0, 1.0, tmap.res_x, tmap.res_y, "./grid.svg", 1.0);
}

void solve(Domain<double>& u, unsigned int max_iter, double tau, std::vector<std::vector<double>>& f_pixels, std::vector<std::vector<double>>& g_pixels) {
    std::vector<std::vector<double>> r = u.u;
    std::vector<std::vector<double>> d = u.u;

    const std::vector<std::vector<int>> dirs = u.generate_directions(8);
    const std::vector<std::pair<int,int>> bases = u.generate_bases(dirs);

    double prev_residual = 0.0;
    
    const int window = 10;
    std::deque<double> residual_history;
    double smoothed_residual = 0.0;

    for (unsigned int iter = 0; iter < max_iter; iter++) {
        double residual = 0.0;

        std::vector<std::vector<double>> new_u = u.u;

        residual = u.compute_residual(f_pixels, g_pixels, dirs, bases, r);
        
        subtractAverage(r);
        poisson_solver(r, d, u.res_x, u.res_y, 1e6, 1e-6, 16);

        for (int x = 0; x < u.res_x; x++) {
            for (int y = 0; y < u.res_y; y++) {
                new_u[x][y] += tau * d[x][y];
            }
        }

        u.u = new_u;

        // --- smoothing update ---
        residual_history.push_back(residual);
        if (residual_history.size() > window) {
            residual_history.pop_front();
        }
        smoothed_residual = std::accumulate(residual_history.begin(), residual_history.end(), 0.0)
                            / residual_history.size();

        // stop if smoothed residual change is small
        if (iter > window) {
            double prev_avg = std::accumulate(residual_history.begin(), residual_history.end() - 1, 0.0)
                            / (residual_history.size() - 1);
        
            std::cout << "Iter " << iter << " max residual = " << residual << ", change = " << prev_avg - smoothed_residual << std::endl;

            if (std::abs(prev_avg - smoothed_residual) < 0.0001) {
                break;
            }
        } else {
            std::cout << "Iter " << iter << " max residual = " << residual << ", change = " << smoothed_residual << std::endl;
        }

        if (iter % 10 == 0) {
            export_transported_pts(u);
        }
    }
}
//*/

int main(int argc, char** argv) {
    std::vector<std::vector<double>> S_MABV2;

    // Parse user arguments
    args::ArgumentParser parser("", "");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

    args::ValueFlag<std::string>    target_png(parser, "image", "Input target image", {"target_png"});
    args::ValueFlag<std::string>    source_png(parser, "image", "Input source image", {"source_png"});
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
    std::string f_image_filename = "";
    std::string g_image_filename = "";

    bool output_progress = false;
    unsigned int res_w = 100;

    if (target_png) { 
        f_image_filename = args::get(target_png);
    }

    if (source_png) { 
        g_image_filename = args::get(source_png);
    }

    if (resolution) {
        res_w = args::get(resolution);
    }

	std::vector<std::vector<double>> f_pixels;
    std::vector<std::vector<double>> g_pixels;
    image_to_grid(f_image_filename.c_str(), f_pixels);
    image_to_grid(g_image_filename.c_str(), g_pixels);

    //pixels = scale_matrix_proportional(pixels, 0.0, 1.0);
    f_pixels = hammingResize(f_pixels, f_pixels[0].size(), res_w);

    normalizeImage(f_pixels, 1.0, 1.0);
    normalizeImage(g_pixels, 1.0, 1.0);

    normal_integration normal_int;
    Domain<double> u(f_pixels[0].size(), f_pixels.size());

    solve(u, 10000, 1.0 / (u.res_x * u.res_y), f_pixels, g_pixels);

    std::vector<std::vector<double>> points;
    Mesh tmap = extract_map(u);
    Mesh mesh(0.0, 1.0, 0.0, 1.0, 128, 128);
    mesh.make_circular();

    std::vector<std::vector<double>> trg_pts = apply_inverse_map(tmap, mesh.source_points);
    export_grid_to_svg(trg_pts, 1.0, 1.0, mesh.res_x, mesh.res_y, "./grid_inv.svg", 1.0);

    mesh.build_vertex_to_triangles();

    normal_int.initialize_data(mesh);

    std::vector<std::vector<double>> desired_normals;

    double focal_l = 1.0;
    double thickness = 0.2;

    //scalePoints(trg_pts, {8, 8, 0}, {0.5, 0.5, 0});
    //rotatePoints(trg_pts, {0, 0, 0});
    translatePoints(trg_pts, {0, 0, -focal_l});

    double r = 1.55;

    mesh.calculate_vertex_laplacians();

    for (int i=0; i<10; i++)
    {
        double max_z = -10000;

        for (int j = 0; j < mesh.source_points.size(); j++) {
            if (max_z < mesh.source_points[j][2]) {
            max_z = mesh.source_points[j][2];
            }
        }

        for (int j = 0; j < mesh.source_points.size(); j++) {
            mesh.source_points[j][2] -= max_z;
        }

        //std::cout << "mesh.source_points.size() = " << mesh.source_points.size() << std::endl;
        //std::cout << "trg_pts.size() = " << trg_pts.size() << std::endl;
        
        std::vector<std::vector<double>> normals = fresnelMapping(mesh.source_points, trg_pts, r);

        normal_int.perform_normal_integration(mesh, normals);
    }

    mesh.save_solid_obj_source(thickness, "./output.obj");
}