/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make
 ============================================================================
 */

// hough-constant.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include "pgm.h"
#include "utils.h"
#include "cpu_hough.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

// Declaración de memoria constante
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

// Kernel de Hough Transform
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;

    int xCent = w / 2;
    int yCent = h / 2;

    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    __shared__ int localAcc[degreeBins * rBins];
    if (threadIdx.x < degreeBins * rBins)
        localAcc[threadIdx.x] = 0;
    __syncthreads();

    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            atomicAdd(&localAcc[rIdx * degreeBins + tIdx], 1);
        }
    }

    __syncthreads();
    if (threadIdx.x < degreeBins * rBins) {
        atomicAdd(&acc[threadIdx.x], localAcc[threadIdx.x]);
    }
}

int main(int argc, char **argv) {
    PGMImage inImg(argv[1]);

    int *cpuht;
    int w = inImg.x_dim;
    int h = inImg.y_dim;

    // Pre-compute and copy cos/sin values
    float *pcCos, *pcSin;
    precomputeTrig(degreeBins, radInc, &pcCos, &pcSin);

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);

    // Ejecutar en la CPU para comparar
    CPU_HoughTran(inImg.pixels, w, h, &cpuht);

    // GPU memory setup
    unsigned char *d_in;
    int *d_hough;
    cudaMalloc((void **) &d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **) &d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, inImg.pixels, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    int blockNum = ceil(w * h / 256.0);
    GPU_HoughTran<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale);

    // Copy results back and check
    int *h_hough = (int *) malloc(degreeBins * rBins * sizeof(int));
    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);
    compareResults(cpuht, h_hough, degreeBins * rBins);

    // Cleanup
    cleanup(d_in, d_hough, pcCos, pcSin, h_hough, cpuht);
    printf("Done!\n");

    return 0;
}
