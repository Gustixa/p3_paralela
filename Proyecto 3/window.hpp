#pragma once

#define _USE_MATH_DEFINES
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <cstdlib>

#include "cpu_hough.hpp"
#include "image.hpp"

#include <QtWidgets>
#include <QtCore>
#include <QtGui>

struct Window : QMainWindow {
	Image input_image;

	QLabel* image_a;
	QLabel* image_b;
	QLabel* image_c;
	QLabel* image_d;

	int THRESHOLD;

	Image cpu_output_image_a;
	Image cpu_output_image_b;
	Image cpu_output_image_overlay_a;
	Image cpu_output_image_overlay_b;

	unsigned char* d_in;
	float* pcCos;
	float* pcSin;
	int* d_hough;
	int* cpuht;
	int* gpuht;

	Window();
	~Window();

	void processCuda();
};