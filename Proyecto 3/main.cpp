/**
 * @file main.cpp
 * @brief Punto de entrada para la aplicaci�n de Transformada de Hough en CPU y GPU usando Qt.
 */
#include "window.hpp"

 /**
  * @brief Funci�n principal de la aplicaci�n.
  *
  * Inicializa la aplicaci�n Qt, crea una instancia de la ventana principal y la muestra en modo maximizado.
  *
  * @param argc N�mero de argumentos de l�nea de comandos.
  * @param argv Argumentos de l�nea de comandos.
  * @return int C�digo de salida de la aplicaci�n.
  */
int main(int argc, char **argv) {
	QApplication* app = new QApplication(argc, argv);
	Window* window = new Window();

	window->showMaximized();
	app->exec();
	
	delete window;
	delete app;
	return 0;
}