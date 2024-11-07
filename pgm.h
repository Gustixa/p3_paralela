// pgm.h
#ifndef PGM_H
#define PGM_H

struct PGMImage {
    int x_dim;
    int y_dim;
    unsigned char *pixels;
    PGMImage(const char *filename);
    ~PGMImage();
};

#endif
