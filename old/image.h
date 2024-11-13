// image.h
#ifndef IMAGE_H
#define IMAGE_H

struct Image {
    int x_dim;
    int y_dim;
    unsigned char *pixels;
    Image(const char *filename);
    ~Image();
};

#endif
