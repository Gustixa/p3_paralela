/**
 * @file utils.cpp
 * @brief Utility functions for precomputing trigonometric values, comparing results, and cleaning up memory.
 */

#include "utils.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

/**
 * @brief Precomputes cosine and sine values for a set of angles.
 *
 * This function calculates cosine and sine values for `degreeBins` angles, with each angle incrementing
 * by `radInc` radians, and stores these values in the provided pointers `pcCos` and `pcSin`.
 *
 * @param degreeBins The number of angle bins to precompute.
 * @param radInc The increment in radians for each successive angle.
 * @param pcCos A pointer to store the array of precomputed cosine values.
 * @param pcSin A pointer to store the array of precomputed sine values.
 */
void precomputeTrig(int degreeBins, float radInc, float **pcCos, float **pcSin) {
    *pcCos = (float *) malloc(sizeof(float) * degreeBins);
    *pcSin = (float *) malloc(sizeof(float) * degreeBins);
    float rad = 0;
    for (int i = 0; i < degreeBins; i++) {
        (*pcCos)[i] = cos(rad);
        (*pcSin)[i] = sin(rad);
        rad += radInc;
    }
}

/**
 * @brief Compares the results of two Hough Transform computations (CPU vs. GPU).
 *
 * This function compares two arrays, `cpuht` and `gpuht`, of size `size`, which
 * contain the Hough Transform results from CPU and GPU computations respectively.
 * If any mismatch is found between the arrays, the index and differing values are printed.
 *
 * @param cpuht Pointer to the CPU-computed Hough Transform results array.
 * @param gpuht Pointer to the GPU-computed Hough Transform results array.
 * @param size The size of the result arrays to compare.
 */
void compareResults(int *cpuht, int *gpuht, int size) {
    for (int i = 0; i < size; i++) {
        if (cpuht[i] != gpuht[i]) {
            printf("Calculation mismatch at: %i %i %i\n", i, cpuht[i], gpuht[i]);
        }
    }
}

/**
 * @brief Frees allocated memory for both host and device data.
 *
 * This function releases memory allocated for device data (`d_in`, `d_hough`) using `cudaFree`
 * and for host data (`pcCos`, `pcSin`, `h_hough`, `cpuht`) using `free`.
 *
 * @param d_in Pointer to device input data (to be freed with `cudaFree`).
 * @param d_hough Pointer to device Hough Transform results (to be freed with `cudaFree`).
 * @param pcCos Pointer to host array of precomputed cosine values.
 * @param pcSin Pointer to host array of precomputed sine values.
 * @param h_hough Pointer to host Hough Transform results array.
 * @param cpuht Pointer to host CPU-computed Hough Transform results array.
 */
void cleanup(unsigned char *d_in, int *d_hough, float *pcCos, float *pcSin, int *h_hough, int *cpuht) {
    cudaFree(d_in);
    cudaFree(d_hough);
    free(pcCos);
    free(pcSin);
    free(h_hough);
    free(cpuht);
}
