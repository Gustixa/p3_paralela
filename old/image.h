/**
 * @file image.h
 * @brief Definition of the Image struct for loading and managing image data.
 */

#ifndef IMAGE_H
#define IMAGE_H

/**
 * @struct Image
 * @brief A struct for loading and managing images.
 *
 * The Image struct provides functionality to load an image from a file,
 * store its dimensions and pixel data, and release memory upon destruction.
 */
struct Image {
    int x_dim;              /**< Width of the image in pixels. */
    int y_dim;              /**< Height of the image in pixels. */
    unsigned char *pixels;  /**< Pointer to the pixel data of the image. */

    /**
     * @brief Constructs an Image object by loading an image from the specified file.
     *
     * This constructor initializes the image by loading its pixel data and setting its
     * dimensions based on the contents of the file.
     *
     * @param filename The path to the image file to load.
     * @throw std::runtime_error if the file cannot be opened or the image cannot be loaded.
     */
    Image(const char *filename);

    /**
     * @brief Destructor for the Image struct.
     *
     * Releases the memory allocated for the image pixel data.
     */
    ~Image();
};

#endif // IMAGE_H
