/**
 * @file cpu_hough.hpp
 * @brief Declaraciones de funciones para la Transformada de Hough en CPU y precomputaci�n de valores trigonom�tricos.
 */
#pragma once

#include "image.hpp"

 /**
  * @brief Realiza la Transformada de Hough en la imagen de entrada.
  *
  * @param input Imagen de entrada en escala de grises sobre la cual se aplicar� la Transformada de Hough.
  * @param output_a Imagen de salida donde se dibujan las l�neas detectadas.
  * @param output_b Imagen de salida donde se dibujan los puntos de intersecci�n de las l�neas.
  * @param output_overlay_a Imagen de salida donde se dibujan las l�neas detectadas superpuestas sobre la imagen original.
  * @param output_overlay_b Imagen de salida donde se dibujan los puntos de intersecci�n superpuestos sobre la imagen original.
  * @param threshold Umbral para considerar una acumulaci�n como una detecci�n v�lida.
  * @param acc Puntero a la matriz de acumulaci�n que almacena las detecciones de l�neas en el espacio de Hough.
  *
  * Esta funci�n aplica la Transformada de Hough para detectar l�neas en la imagen de entrada y almacena los resultados
  * en las im�genes de salida y en la matriz de acumulaci�n `acc`.
  */
void CPU_HoughTran(const Image& input, Image& output_a, Image& output_b, Image& output_overlay_a, Image& output_overlay_b, const int& threshold, int **acc);

/**
 * @brief Precalcula valores trigonom�tricos para la Transformada de Hough.
 *
 * @param degreeBins N�mero de divisiones angulares para la Transformada de Hough.
 * @param radInc Incremento en radianes para cada divisi�n angular.
 * @param pcCos Puntero al arreglo donde se almacenar�n los valores precalculados de coseno.
 * @param pcSin Puntero al arreglo donde se almacenar�n los valores precalculados de seno.
 *
 * Esta funci�n precalcula y almacena los valores de seno y coseno para cada �ngulo en la Transformada de Hough,
 * lo que permite una ejecuci�n m�s r�pida al evitar c�lculos redundantes.
 */
void precomputeTrig(int degreeBins, float radInc, float **pcCos, float **pcSin);