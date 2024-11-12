#include "image.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image_write.h"
#include "stb_image.h"
#include <stdexcept>

Image::Image() {
	width = 0;
	height = 0;
}

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

void Image::saveImage(const char* filename) {
	if (!stbi_write_png(filename, width, height, STBI_grey, pixels.data(), width)) {
		printf("Failed to write the image.");
		exit(101);
	}
}

Image::Image(const Image& other) {
	width = other.width;
	height = other.height;
	pixels = std::vector<unsigned char>(width * height, 0);
}

Image Image::copy(const Image& other) {
	Image image = other;
	for (int i = 0; i <image.width * image.height; ++i) {
		image.pixels[i] = other.pixels[i];
	}
	return image;
}

Image Image::paramCopy(const Image& other) {
	Image image;
	image.width = other.width;
	image.height = other.height;
	image.pixels = std::vector<unsigned char>(image.width * image.height, 0);
	return image;
}