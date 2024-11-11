// cpu_hough.cpp
#include "cpu_hough.h"
#include <cmath>
#include <cstring>

void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc) {
    const int degreeInc = 2;
    const int degreeBins = 180 / degreeInc;
    const int rBins = 100;
    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    *acc = new int[rBins * degreeBins];
    memset(*acc, 0, sizeof(int) * rBins * degreeBins);

    int xCent = w / 2;
    int yCent = h / 2;
    float rScale = 2 * rMax / rBins;
    float radInc = degreeInc * M_PI / 180;

    for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++) {
            int idx = j * w + i;
            if (pic[idx] > 0) {
                int xCoord = i - xCent;
                int yCoord = yCent - j;
                float theta = 0;
                for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
                    float r = xCoord * cos(theta) + yCoord * sin(theta);
                    int rIdx = (r + rMax) / rScale;
                    (*acc)[rIdx * degreeBins + tIdx]++;
                    theta += radInc;
                }
            }
        }
}
