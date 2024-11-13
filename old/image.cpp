// image.cpp
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "image.h"
#include <stdexcept>  // Agregar esta línea

Image::Image(const char* filename) {
    int channels;  // Definir la variable channels aquí
    pixels = stbi_load(filename, &x_dim, &y_dim, &channels, 0);
    if (!pixels) {
        throw std::runtime_error("Error cargando la imagen");
    }
}

Image::~Image() {
    stbi_image_free(pixels);
}
