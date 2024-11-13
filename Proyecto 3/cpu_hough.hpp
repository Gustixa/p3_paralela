#pragma once

#include "image.hpp"

void CPU_HoughTran(const Image& input, Image& output_a, Image& output_b, Image& output_overlay_a, Image& output_overlay_b, const int& threshold, int **acc);
void precomputeTrig(int degreeBins, float radInc, float **pcCos, float **pcSin);