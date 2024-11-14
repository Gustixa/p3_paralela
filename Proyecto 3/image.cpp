/**
 * @file image.cpp
 * @brief Implementación de la clase Image para cargar, guardar y manipular imágenes en escala de grises.
 */
#include "image.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image_write.h"
#include "stb_image.h"
#include <stdexcept>

 /**
  * @brief Constructor por defecto de la clase Image.
  *
  * Inicializa la imagen con dimensiones 0x0.
  */
Image::Image() {
	width = 0;
	height = 0;
}

/**
 * @brief Constructor que carga una imagen en escala de grises desde un archivo.
 *
 * @param filename Nombre del archivo de imagen a cargar.
 *
 * Carga la imagen en escala de grises utilizando la biblioteca stb_image y almacena los píxeles en el vector `pixels`.
 * Si la carga falla, se imprime un mensaje de error y se termina el programa con un código de error 100.
 */
Image::Image(const char *filename) {
	int channels;
	unsigned char* data = stbi_load(filename, &width, &height, &channels, STBI_grey);
	if (!data) {
		printf("Failed to load the image.");
		exit(100);
	}

	for (int i = 0; i < width * height; i++) {
		pixels.push_back(data[i]);
	}
	stbi_image_free(data);
}

/**
 * @brief Guarda la imagen en un archivo PNG.
 *
 * @param filename Nombre del archivo donde se guardará la imagen.
 *
 * Guarda la imagen en formato PNG usando stb_image_write. Si la escritura falla, se imprime un mensaje de error
 * y se termina el programa con un código de error 101.
 */
void Image::saveImage(const char* filename) {
	if (!stbi_write_png(filename, width, height, STBI_grey, pixels.data(), width)) {
		printf("Failed to write the image.");
		exit(101);
	}
}

/**
 * @brief Constructor de copia de la clase Image.
 *
 * @param other Objeto Image a copiar.
 *
 * Crea una copia de la imagen dada, copiando las dimensiones de `other` pero inicializando los píxeles a cero.
 */
Image::Image(const Image& other) {
	width = other.width;
	height = other.height;
	pixels = std::vector<unsigned char>(width * height, 0);
}

/**
 * @brief Realiza una copia profunda de una imagen.
 *
 * @param other Objeto Image a copiar.
 * @return Image Nueva instancia de Image que es una copia de `other`.
 *
 * Crea una copia exacta de la imagen, copiando todos los píxeles de `other` en la nueva instancia.
 */
Image Image::copy(const Image& other) {
	Image image = other;
	for (int i = 0; i <image.width * image.height; ++i) {
		image.pixels[i] = other.pixels[i];
	}
	return image;
}

/**
 * @brief Crea una copia con parámetros de una imagen.
 *
 * @param other Objeto Image a copiar.
 * @return Image Nueva instancia de Image con las mismas dimensiones que `other`, pero con píxeles inicializados a cero.
 *
 * Crea una nueva instancia de `Image` con las mismas dimensiones que `other`, pero sin copiar el contenido de los píxeles.
 */
Image Image::paramCopy(const Image& other) {
	Image image;
	image.width = other.width;
	image.height = other.height;
	image.pixels = std::vector<unsigned char>(image.width * image.height, 0);
	return image;
}