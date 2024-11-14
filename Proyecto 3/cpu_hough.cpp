/**
 * @file cpu_hough.hpp
 * @brief Implementación de la Transformada de Hough en la CPU.
 */

#include "cpu_hough.hpp"
#include <cstring>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <vector>

 /**
  * @brief Realiza la Transformada de Hough en la CPU para detectar líneas y puntos en una imagen binaria.
  *
  * Esta función utiliza la Transformada de Hough para detectar líneas y puntos en una imagen de entrada.
  * Los resultados se almacenan en imágenes de salida, y la función permite superponer las detecciones.
  *
  * @param input Imagen de entrada, debe estar en formato binario (0 o 255 por pixel).
  * @param output_a Imagen de salida para las líneas detectadas.
  * @param output_b Imagen de salida para los puntos detectados.
  * @param output_overlay_a Imagen de superposición para líneas detectadas.
  * @param output_overlay_b Imagen de superposición para puntos detectados.
  * @param threshold Umbral de acumulación para considerar una línea o punto detectado.
  * @param acc Puntero a un acumulador donde se almacena el conteo de (r, theta) para cada punto de la imagen.
  */
void CPU_HoughTran(const Image& input, Image& output_a, Image& output_b, Image& output_overlay_a, Image& output_overlay_b, const int& threshold, int** acc) {
	const int degreeInc = 2;
	const int degreeBins = 180 / degreeInc;
	const int rBins = 100;
	const float rMax = std::sqrt(1.0f * input.width * input.width + 1.0f * input.height * input.height) / 2.0f;
	*acc = new int[rBins * degreeBins];
	std::vector<float> values = std::vector<float>(rBins * degreeBins, 0.0f);
	memset(*acc, 0, sizeof(int) * rBins * degreeBins);

	const int xCent = input.width / 2;
	const int yCent = input.height / 2;
	const float rScale = 2 * rMax / rBins;
	const float radInc = (float)(degreeInc * M_PI / 180.0);

	int maxDist = static_cast<int>(std::sqrt(input.width * input.width + input.height * input.height));
	int angleSteps = 180; // Angles from -90 to +90 degrees
	std::vector<std::vector<int>> accumulator(2 * maxDist, std::vector<int>(angleSteps, 0));

	{
		int x = 0;
		int y = 0;
		int size_x = input.width;
		int size_y = input.height;
		#pragma omp parallel for private(x, y) collapse(2) num_threads(12)
		for (x = 0; x < size_x; x++) {
			for (y = 0; y < size_y; y++) {
				const int idx = y * input.width + x;
				if (input.pixels[idx] > 0) {
					const int xCoord = x - xCent;
					const int yCoord = yCent - y;
					float theta = 0;
					for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
						const float r = xCoord * std::cos(theta) + yCoord * std::sin(theta);
						const int rIdx = static_cast<int>((r + rMax) / rScale);
						(*acc)[rIdx * degreeBins + tIdx]++;
						theta += radInc;
						values[rIdx * degreeBins + tIdx] += radInc;
					}
					for (int theta = 0; theta < angleSteps; ++theta) {
						double thetaRad = theta * M_PI / 180.0;
						int r = static_cast<int>(x * std::cos(thetaRad) + y * std::sin(thetaRad)) + maxDist;
						if (r >= 0 && r < 2 * maxDist) {
							accumulator[r][theta]++;
						}
					}
				}
			}
		}
	}

	// Output Lineas
	for (int r = 0; r < 2 * maxDist; ++r) {
		for (int theta = 0; theta < angleSteps; ++theta) {
			if (accumulator[r][theta] > threshold) {
				// Draw the line in the output image based on r and theta
				double thetaRad = theta * M_PI / 180.0;
				for (int x = 0; x < input.width; ++x) {
					int y = static_cast<int>((r - maxDist - x * std::cos(thetaRad)) / std::sin(thetaRad));
					if (y >= 0 && y < input.height) {
						output_a.pixels[y * output_a.width + x] = 255; // White line
						output_overlay_a.pixels[y * output_a.width + x] = 255; // White line
					}
				}
			}
		}
	}

	// Output Puntos
	for (int rIdx = 0; rIdx < rBins; rIdx++) {
		for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
			if ((*acc)[rIdx * degreeBins + tIdx] >= threshold) {
				// Convert polar coordinates (r, theta) to Cartesian coordinates
				float theta = tIdx * radInc;
				float r = rIdx * rScale - rMax;

				// Find (x, y) positions corresponding to this (r, theta)
				int x = static_cast<int>(r * std::cos(theta)) + xCent;
				int y = yCent - static_cast<int>(r * std::sin(theta));

				// Check bounds and set pixel to white if within the image
				if (x >= 0 && x < output_b.width && y >= 0 && y < output_b.height) {
					output_b.pixels[y * output_b.width + x] = 255; // Set to white
					output_overlay_b.pixels[y * output_b.width + x] = 255; // Set to white
				}
			}
		}
	}
}

/**
 * @brief Precalcula los valores de seno y coseno para optimizar la Transformada de Hough.
 *
 * Esta función almacena en arreglos los valores precalculados de seno y coseno para cada ángulo
 * en los bins de grados, con el fin de optimizar el cálculo durante la Transformada de Hough.
 *
 * @param degreeBins Número de bins de ángulos en grados.
 * @param radInc Incremento en radianes por cada ángulo.
 * @param pcCos Puntero donde se almacenará el arreglo precalculado de cosenos.
 * @param pcSin Puntero donde se almacenará el arreglo precalculado de senos.
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