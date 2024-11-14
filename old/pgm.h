/**
 * @file pgm.h
 * @brief Definition of the PGMImage struct for handling grayscale PGM images.
 */

#ifndef PGM_H
#define PGM_H

/**
 * @struct PGMImage
 * @brief A struct for loading and managing grayscale PGM images in binary (P5) format.
 *
 * The PGMImage struct provides functionality to load a grayscale PGM (P5) image from a file,
 * store its dimensions and pixel data, and release memory upon destruction.
 */
struct PGMImage {
    int x_dim;              /**< Width of the image in pixels. */
    int y_dim;              /**< Height of the image in pixels. */
    unsigned char *pixels;  /**< Pointer to the pixel data of the image. */

    /**
     * @brief Constructs a PGMImage object by loading a PGM (P5) image from the specified file.
     *
     * This constructor initializes the image by loading pixel data and setting dimensions
     * based on the contents of the PGM file.
     *
     * @param filename The path to the PGM image file to load.
     * @throw std::runtime_error if the file cannot be opened or is not a valid PGM (P5) image.
     */
    PGMImage(const char *filename);

    /**
     * @brief Destructor for the PGMImage struct.
     *
     * Releases the memory allocated for the image pixel data.
     */
    ~PGMImage();
};

#endif // PGM_H
