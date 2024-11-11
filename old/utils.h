// utils.h
#ifndef UTILS_H
#define UTILS_H

void precomputeTrig(int degreeBins, float radInc, float **pcCos, float **pcSin);
void compareResults(int *cpuht, int *gpuht, int size);
void cleanup(unsigned char *d_in, int *d_hough, float *pcCos, float *pcSin, int *h_hough, int *cpuht);

#endif
