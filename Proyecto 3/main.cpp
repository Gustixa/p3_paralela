/**
 * @file main.cpp
 * @brief Punto de entrada para la aplicación de Transformada de Hough en CPU y GPU usando Qt.
 */
#include "window.hpp"

 /**
  * @brief Función principal de la aplicación.
  *
  * Inicializa la aplicación Qt, crea una instancia de la ventana principal y la muestra en modo maximizado.
  *
  * @param argc Número de argumentos de línea de comandos.
  * @param argv Argumentos de línea de comandos.
  * @return int Código de salida de la aplicación.
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