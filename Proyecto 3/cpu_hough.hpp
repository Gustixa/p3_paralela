/**
 * @file cpu_hough.hpp
 * @brief Declaraciones de funciones para la Transformada de Hough en CPU y precomputación de valores trigonométricos.
 */
#pragma once

#include "image.hpp"

 /**
  * @brief Realiza la Transformada de Hough en la imagen de entrada.
  *
  * @param input Imagen de entrada en escala de grises sobre la cual se aplicará la Transformada de Hough.
  * @param output_a Imagen de salida donde se dibujan las líneas detectadas.
  * @param output_b Imagen de salida donde se dibujan los puntos de intersección de las líneas.
  * @param output_overlay_a Imagen de salida donde se dibujan las líneas detectadas superpuestas sobre la imagen original.
  * @param output_overlay_b Imagen de salida donde se dibujan los puntos de intersección superpuestos sobre la imagen original.
  * @param threshold Umbral para considerar una acumulación como una detección válida.
  * @param acc Puntero a la matriz de acumulación que almacena las detecciones de líneas en el espacio de Hough.
  *
  * Esta función aplica la Transformada de Hough para detectar líneas en la imagen de entrada y almacena los resultados
  * en las imágenes de salida y en la matriz de acumulación `acc`.
  */
void CPU_HoughTran(const Image& input, Image& output_a, Image& output_b, Image& output_overlay_a, Image& output_overlay_b, const int& threshold, int **acc);

/**
 * @brief Precalcula valores trigonométricos para la Transformada de Hough.
 *
 * @param degreeBins Número de divisiones angulares para la Transformada de Hough.
 * @param radInc Incremento en radianes para cada división angular.
 * @param pcCos Puntero al arreglo donde se almacenarán los valores precalculados de coseno.
 * @param pcSin Puntero al arreglo donde se almacenarán los valores precalculados de seno.
 *
 * Esta función precalcula y almacena los valores de seno y coseno para cada ángulo en la Transformada de Hough,
 * lo que permite una ejecución más rápida al evitar cálculos redundantes.
 */
void precomputeTrig(int degreeBins, float radInc, float **pcCos, float **pcSin);