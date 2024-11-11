// utils.cpp
#include "utils.h"
#include <cuda_runtime_api.h>  // Usa cuda_runtime_api.h en lugar de cuda_runtime.h
#include <cmath>
#include <cstdio>
#include <cstdlib>

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

void compareResults(int *cpuht, int *gpuht, int size) {
    for (int i = 0; i < size; i++) {
        if (cpuht[i] != gpuht[i]) {
            printf("Calculation mismatch at: %i %i %i\n", i, cpuht[i], gpuht[i]);
        }
    }
}

void cleanup(unsigned char *d_in, int *d_hough, float *pcCos, float *pcSin, int *h_hough, int *cpuht) {
    cudaFree(d_in);
    cudaFree(d_hough);
    free(pcCos);
    free(pcSin);
    free(h_hough);
    free(cpuht);
}
