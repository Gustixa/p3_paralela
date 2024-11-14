/**
 * @file utils.h
 * @brief Declaration of utility functions for trigonometric precomputation, result comparison, and memory cleanup.
 */

#ifndef UTILS_H
#define UTILS_H

/**
 * @brief Precomputes arrays of cosine and sine values for a set of angles.
 *
 * This function allocates and fills arrays for the cosine and sine values
 * of angles incrementing by `radInc` radians, up to `degreeBins` bins.
 *
 * @param degreeBins The number of angle bins to precompute.
 * @param radInc The increment in radians for each successive angle.
 * @param pcCos Pointer to store the array of precomputed cosine values.
 * @param pcSin Pointer to store the array of precomputed sine values.
 */
void precomputeTrig(int degreeBins, float radInc, float **pcCos, float **pcSin);

/**
 * @brief Compares two arrays containing Hough Transform results.
 *
 * This function compares the results from CPU and GPU Hough Transform computations.
 * It prints an error message for any mismatched elements, indicating the index and differing values.
 *
 * @param cpuht Pointer to the CPU-computed Hough Transform results array.
 * @param gpuht Pointer to the GPU-computed Hough Transform results array.
 * @param size The size of the result arrays to compare.
 */
void compareResults(int *cpuht, int *gpuht, int size);

/**
 * @brief Frees allocated memory for both host and device arrays.
 *
 * This function releases memory for device data (`d_in`, `d_hough`) using `cudaFree`
 * and for host arrays (`pcCos`, `pcSin`, `h_hough`, `cpuht`) using `free`.
 *
 * @param d_in Pointer to device input data (to be freed with `cudaFree`).
 * @param d_hough Pointer to device Hough Transform results (to be freed with `cudaFree`).
 * @param pcCos Pointer to host array of precomputed cosine values.
 * @param pcSin Pointer to host array of precomputed sine values.
 * @param h_hough Pointer to host Hough Transform results array.
 * @param cpuht Pointer to host CPU-computed Hough Transform results array.
 */
void cleanup(unsigned char *d_in, int *d_hough, float *pcCos, float *pcSin, int *h_hough, int *cpuht);

#endif // UTILS_H
