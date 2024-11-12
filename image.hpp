#pragma once

#include <vector>

struct Image {
	int width;
	int height;
	std::vector<unsigned char> pixels;

	Image();
	Image(const char *filename);
	void saveImage(const char* filename);

	Image(const Image& other);

	static Image copy(const Image& other);
	static Image paramCopy(const Image& other);
};