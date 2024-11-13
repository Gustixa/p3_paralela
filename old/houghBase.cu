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
#include "image.h" // Cambiamos pgm.h a image.h
#include "utils.h"
#include "cpu_hough.h"
#include "stb_image.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

// Declaración de memoria constante
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

// Kernel con sólo memoria global
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

    __shared__ int localAcc[degreeBins * rBins];  // Memoria compartida para el acumulador
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


void runExperiment(int kernelType, unsigned char *d_in, int *d_hough, int w, int h, float rMax, float rScale, int numTrials) {
    std::vector<float> times;
    for (int i = 0; i < numTrials; i++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start); // Inicia el temporizador
        
        // Selecciona y ejecuta el kernel basado en kernelType
        if (kernelType == 0) {
            GPU_HoughTran_Global<<<ceil(w * h / 256.0), 256>>>(d_in, w, h, d_hough, rMax, rScale);
        } else if (kernelType == 1) {
            GPU_HoughTran_ConstGlobal<<<ceil(w * h / 256.0), 256>>>(d_in, w, h, d_hough, rMax, rScale);
        } else if (kernelType == 2) {
            GPU_HoughTran_ConstGlobalShared<<<ceil(w * h / 256.0), 256>>>(d_in, w, h, d_hough, rMax, rScale);
        }
        
        cudaEventRecord(stop); // Detiene el temporizador
        cudaEventSynchronize(stop); // Sincroniza para asegurarse de que el kernel ha terminado

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop); // Calcula el tiempo de ejecución
        times.push_back(milliseconds); // Guarda el tiempo en la lista

        cudaEventDestroy(start); // Limpia los eventos de CUDA
        cudaEventDestroy(stop);
    }
    
    // Calcula el tiempo promedio
    float total = 0;
    for (float time : times) {
        total += time;
    }
    float avgTime = total / numTrials;
    
    // Imprime los tiempos y el promedio
    std::cout << "Kernel Type " << kernelType << ": Average Time = " << avgTime << " ms\n";
    for (int i = 0; i < times.size(); i++) {
        std::cout << "  Trial " << i + 1 << ": " << times[i] << " ms\n";
    }
}


int main(int argc, char **argv) {
    Image inImg(argv[1]);

    int w = inImg.x_dim;
    int h = inImg.y_dim;

    // Pre-compute and copy cos/sin values
    float *pcCos, *pcSin;
    precomputeTrig(degreeBins, radInc, &pcCos, &pcSin);

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);

    // GPU memory setup
    unsigned char *d_in;
    int *d_hough;
    cudaMalloc((void **) &d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **) &d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, inImg.pixels, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    int numTrials = 10;
    
    std::cout << "Running experiments...\n";
    
    // Ejecutar experimentos para cada configuración de memoria
    runExperiment(0, d_in, d_hough, w, h, rMax, rScale, numTrials); // Solo Global
    runExperiment(1, d_in, d_hough, w, h, rMax, rScale, numTrials); // Constante y Global
    runExperiment(2, d_in, d_hough, w, h, rMax, rScale, numTrials); // Constante, Global y Compartida

    // Cleanup
    cleanup(d_in, d_hough, pcCos, pcSin, nullptr, nullptr);
    printf("Done!\n");

    return 0;
}