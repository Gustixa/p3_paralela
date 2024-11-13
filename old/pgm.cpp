/**
 * @file pgm.cpp
 * @brief Implementation of the PGMImage class for loading PGM (P5) images.
 */

#include "pgm.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <limits>

/**
 * @class PGMImage
 * @brief A class for loading and managing grayscale PGM images in binary (P5) format.
 *
 * The PGMImage class provides functionality to load a PGM (P5) image from a file,
 * handle its pixel data, and release memory upon destruction.
 */

/**
 * @brief Constructs a PGMImage object by loading a PGM (P5) image from the specified file.
 *
 * This constructor opens a PGM file in binary mode and reads its header to verify
 * the format, then loads the image dimensions and pixel data. If the file cannot
 * be opened or if it is not a valid PGM (P5) image, an exception is thrown.
 *
 * @param filename The path to the PGM image file to load.
 * @throw std::runtime_error if the file cannot be opened, is not in P5 format, or contains unsupported data.
 */
PGMImage::PGMImage(const char *filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo PGM.");
    }

    std::string header;
    int maxVal;

    // Leer el encabezado del archivo PGM
    file >> header;
    if (header != "P5") {
        throw std::runtime_error("Formato PGM no soportado (se requiere formato P5).");
    }

    // Ignorar comentarios (líneas que comienzan con #)
    file.ignore(); // Ignorar el espacio después del header
    while (file.peek() == '#') {
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    // Leer las dimensiones de la imagen
    file >> x_dim >> y_dim;
    file >> maxVal;
    file.ignore(); // Ignorar el salto de línea después del maxVal

    if (maxVal > 255) {
        throw std::runtime_error("Se espera un valor máximo de 255 para imágenes PGM en 8 bits.");
    }

    // Reservar memoria para los píxeles
    pixels = new unsigned char[x_dim * y_dim];

    // Leer los datos de la imagen
    file.read(reinterpret_cast<char*>(pixels), x_dim * y_dim);

    if (!file) {
        delete[] pixels;
        throw std::runtime_error("Error al leer los datos de la imagen.");
    }

    file.close();
}

/**
 * @brief Destructor for the PGMImage class.
 *
 * This destructor releases the memory allocated for the image pixel data.
 */
PGMImage::~PGMImage() {
    delete[] pixels;
}
