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
// houghBase.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <chrono> // Librería para medir tiempos en CPU
#include "image.h"
#include "utils.h"
#include "stb_image.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

// Kernel con solo memoria global
__global__ void GPU_HoughTran_Global(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;

    int xCent = w / 2;
    int yCent = h / 2;
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            atomicAdd(&acc[rIdx * degreeBins + tIdx], 1);
        }
    }
}

// Kernel con memoria constante y global
__global__ void GPU_HoughTran_ConstGlobal(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;

    int xCent = w / 2;
    int yCent = h / 2;
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            atomicAdd(&acc[rIdx * degreeBins + tIdx], 1);
        }
    }
}

// Kernel con memoria constante, global y compartida
__global__ void GPU_HoughTran_ConstGlobalShared(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale) {
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

void CPU_HoughTransform(unsigned char *pic, int w, int h, int *acc, float *cosTable, float *sinTable, float rMax, float rScale) {
    int xCent = w / 2;
    int yCent = h / 2;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int index = y * w + x;
            if (pic[index] > 0) {
                int xCoord = x - xCent;
                int yCoord = yCent - y;
                for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
                    float r = xCoord * cosTable[tIdx] + yCoord * sinTable[tIdx];
                    int rIdx = (r + rMax) / rScale;
                    acc[rIdx * degreeBins + tIdx]++;
                }
            }
        }
    }
}

void runExperiment(int kernelType, unsigned char *d_in, int *d_hough, int w, int h, float rMax, float rScale, int numTrials) {
    std::vector<float> times;
    for (int i = 0; i < numTrials; i++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);

        if (kernelType == 0) {
            GPU_HoughTran_Global<<<ceil(w * h / 256.0), 256>>>(d_in, w, h, d_hough, rMax, rScale);
        } else if (kernelType == 1) {
            GPU_HoughTran_ConstGlobal<<<ceil(w * h / 256.0), 256>>>(d_in, w, h, d_hough, rMax, rScale);
        } else if (kernelType == 2) {
            GPU_HoughTran_ConstGlobalShared<<<ceil(w * h / 256.0), 256>>>(d_in, w, h, d_hough, rMax, rScale);
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        times.push_back(milliseconds);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    float total = 0;
    for (float time : times) {
        total += time;
    }
    float avgTime = total / numTrials;

    std::cout << "Kernel Type " << kernelType << ": Average Time = " << avgTime << " ms\n";
    for (int i = 0; i < times.size(); i++) {
        std::cout << "  Trial " << i + 1 << ": " << times[i] << " ms\n";
    }
}


int main(int argc, char **argv) {
    Image inImg(argv[1]);

    int w = inImg.x_dim;
    int h = inImg.y_dim;

    float *pcCos, *pcSin;
    precomputeTrig(degreeBins, radInc, &pcCos, &pcSin);

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);

    unsigned char *d_in;
    int *d_hough;
    cudaMalloc((void **) &d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **) &d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, inImg.pixels, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    int numTrials = 10;

    std::cout << "Running experiments...\n";
    runExperiment(0, d_in, d_hough, w, h, rMax, rScale, numTrials);
    runExperiment(1, d_in, d_hough, w, h, rMax, rScale, numTrials);
    runExperiment(2, d_in, d_hough, w, h, rMax, rScale, numTrials);

    int *cpu_hough = (int *)calloc(degreeBins * rBins, sizeof(int));
    auto cpu_start = std::chrono::high_resolution_clock::now();
    CPU_HoughTransform(inImg.pixels, w, h, cpu_hough, pcCos, pcSin, rMax, rScale);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU Hough Transform Time: " << cpu_duration.count() << " ms\n";

    int *gpu_hough = (int *)malloc(sizeof(int) * degreeBins * rBins);
    cudaMemcpy(gpu_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    // compareResults(cpu_hough, gpu_hough, degreeBins * rBins);

    cleanup(d_in, d_hough, pcCos, pcSin, cpu_hough, gpu_hough);
    printf("Done!\n");

    return 0;
}
