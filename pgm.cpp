// pgm.cpp
#include "pgm.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <limits>  // Agregar esta línea

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

PGMImage::~PGMImage() {
    delete[] pixels;
}
