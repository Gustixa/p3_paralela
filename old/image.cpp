/**
 * @file image.cpp
 * @brief Implementation of the Image class for loading and managing images.
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "image.h"
#include <stdexcept>

/**
 * @class Image
 * @brief A class for loading and managing images.
 *
 * The Image class provides functionality to load an image from a file
 * and manage its pixel data, dimensions, and memory cleanup.
 */

/**
 * @brief Constructs an Image object by loading an image from the specified file.
 *
 * This constructor loads an image file using the stb_image library. The image
 * data is stored in `pixels`, and its dimensions are saved in `x_dim` and `y_dim`.
 * If loading the image fails, an exception is thrown.
 *
 * @param filename The path to the image file to load.
 * @throw std::runtime_error if the image cannot be loaded.
 */
Image::Image(const char* filename) {
    int channels;  // Number of color channels in the loaded image.
    pixels = stbi_load(filename, &x_dim, &y_dim, &channels, 0);
    if (!pixels) {
        throw std::runtime_error("Error cargando la imagen");
    }
}

/**
 * @brief Destructor for the Image class.
 *
 * This destructor frees the memory allocated for the image pixel data
 * using `stbi_image_free`.
 */
Image::~Image() {
    stbi_image_free(pixels);
}
