/**
 * @file cpu_hough.cpp
 * @brief Implementation of the Hough Transform on CPU.
 */
#include "cpu_hough.h"
#include <cmath>
#include <cstring>

/**
 * @brief Applies the Hough Transform to detect lines in a binary image.
 *
 * This function performs a Hough Transform on a binary image represented
 * by `pic`, with dimensions `w` (width) and `h` (height). It populates
 * the accumulator array `acc` with the results, where each cell in `acc`
 * represents a (r, θ) pair in polar coordinates, indicating the presence
 * of lines in the image.
 *
 * @param pic A pointer to the binary image data (1D array of unsigned char).
 * @param w The width of the image.
 * @param h The height of the image.
 * @param acc A pointer to the 2D accumulator array for storing Hough Transform results.
 *            The array will be allocated within the function, with `rBins * degreeBins` elements.
 */
void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc) {
    const int degreeInc = 2; // Increment of degrees for θ.
    const int degreeBins = 180 / degreeInc;
    const int rBins = 100;
    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    *acc = new int[rBins * degreeBins];
    memset(*acc, 0, sizeof(int) * rBins * degreeBins);

    int xCent = w / 2;
    int yCent = h / 2;
    float rScale = 2 * rMax / rBins;
    float radInc = degreeInc * M_PI / 180;

    // Iterate over each pixel in the image.
    for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++) {
            int idx = j * w + i;
            if (pic[idx] > 0) { // Process only non-zero pixels (assumes binary image).
                int xCoord = i - xCent;  // X-coordinate relative to center.
                int yCoord = yCent - j;  // Y-coordinate relative to center.
                float theta = 0;

                // Iterate over each angle θ.
                for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
                    float r = xCoord * cos(theta) + yCoord * sin(theta); // Compute radius for (x, y).
                    int rIdx = (r + rMax) / rScale;  // Map radius to bin index.
                    // Increment the accumulator cell for (rIdx, tIdx).
                    (*acc)[rIdx * degreeBins + tIdx]++;
                    theta += radInc; // Move to the next angle.
                }
            }
        }
}
