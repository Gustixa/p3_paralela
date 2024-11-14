/**
 * @file cpu_hough.h
 * @brief Declaration of the CPU-based Hough Transform function.
 */

#ifndef CPU_HOUGH_H
#define CPU_HOUGH_H

/**
 * @brief Performs the Hough Transform on a grayscale image using the CPU.
 *
 * This function applies the Hough Transform to detect lines in a given grayscale image.
 * It takes the pixel data of the image and populates an accumulator array with the results.
 *
 * @param pic Pointer to the grayscale image data in unsigned char format.
 * @param w The width of the image in pixels.
 * @param h The height of the image in pixels.
 * @param acc Pointer to an integer array for storing the accumulator values. 
 *        This array will be dynamically allocated within the function and should
 *        be freed by the caller.
 */
void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc);

#endif // CPU_HOUGH_H
